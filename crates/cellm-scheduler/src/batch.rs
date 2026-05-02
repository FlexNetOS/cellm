// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Continuous batching lite for cellm.
//!
//! On mobile, the typical session count is 2–4. True continuous batching
//! (vLLM-style, with dynamic batch composition) is overkill and introduces
//! scheduling complexity that costs more than it saves.
//!
//! Instead, **continuous batching lite** batches decode steps for sessions
//! that are ready at the same scheduling tick. The key insight is that
//! decode-phase matmuls (QKV projection, output projection, MLP) use the
//! **same model weights** across all sessions. By running those matmuls once
//! with a batch dimension, we amortize weight-fetch overhead and improve
//! hardware utilization — especially on Metal where command-buffer submission
//! latency dominates single-token decode cost.
//!
//! The batching unit is the **decode step**: N sessions each produce one
//! next token, attention runs per-session (different KV context), and the
//! weight-bound matmuls are batched.
//!
//! ## Architecture
//!
//! ```text
//! ┌─┐  ┌─┐  ┌─┐
//! │ Session A │  │ Session B │  │ Session C │
//! │ token=42  │  │ token=7   │  │ token=99  │
//! │ pos=128   │  │ pos=56    │  │ pos=200   │
//! └─┬┘  └─┬┘  └─┬┘
//!      │              │              │
//!      └┼┘
//!                     │
//!          ┌─▼─┐
//!          │   BatchDetector     │  ← checks compatibility
//!          │   groups=[A,B,C]    │
//!          └─┬─┘
//!                     │
//!          ┌─▼─┐
//!          │  Batched Decode     │
//!          │  ┌─┐ │
//!          │  │ QKV matmul     │ │  ← batched (batch_dim=3)
//!          │  │ per-session    │ │
//!          │  │   attention    │ │  ← unbatched (different KV)
//!          │  │ output matmul  │ │  ← batched
//!          │  │ MLP matmuls    │ │  ← batched
//!          │  │ logits split   │ │  ← per-session
//!          │  └─┘ │
//!          └─┬─┘
//!                     │
//!      ┌┼┐
//!      │              │              │
//! ┌─▼┐  ┌─▼┐  ┌─▼┐
//! │ token=X  │  │ token=Y  │  │ token=Z  │
//! │ write KV │  │ write KV │  │ write KV │
//! └─┘  └─┘  └─┘
//! ```

use std::collections::HashMap;

use crate::rr::SessionId;

// BatchGroup─

/// A group of sessions whose decode steps can be batched together.
///
/// Sessions are batch-compatible when they share the same model architecture
/// and have the same hidden_dim / num_heads / head_dim. On mobile, since all
/// sessions use the same loaded model, compatibility is always true — the
/// only constraint is that sessions must be in the Decoding state.
#[derive(Debug, Clone)]
pub struct BatchGroup {
    /// Session IDs in this batch, in decode order.
    pub session_ids: Vec<SessionId>,
    /// Number of tokens to decode per session (1 for decode-phase batching).
    pub tokens_per_session: usize,
}

impl BatchGroup {
    pub fn new(session_ids: Vec<SessionId>) -> Self {
        Self {
            session_ids,
            tokens_per_session: 1,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.session_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.session_ids.is_empty()
    }
}

// BatchDetector 

/// Scans the set of active sessions and groups them into batch-compatible
/// cohorts.
///
/// For the mobile use case (2–4 sessions, single model), compatibility
/// reduces to: is the session in Decoding state and does it have a current
/// token? If yes, it can be batched.
#[derive(Debug, Default)]
pub struct BatchDetector {
    /// Minimum batch size to trigger batching (avoids overhead for single-session).
    pub min_batch_size: usize,
}

impl BatchDetector {
    pub fn new() -> Self {
        Self { min_batch_size: 2 }
    }

    /// Set the minimum number of sessions required before batching kicks in.
    /// Setting this to 1 means every decode tick will attempt to batch
    /// (which is just single-session decode when only one session is active).
    pub fn with_min_batch_size(mut self, n: usize) -> Self {
        self.min_batch_size = n;
        self
    }

    /// Group a set of decode-ready session IDs into batch groups.
    ///
    /// `session_states` maps SessionId → (is_decoding, has_current_token).
    /// Sessions that are not in Decoding state or have no current token
    /// are filtered out.
    ///
    /// On mobile (N ≤ 4), all eligible sessions form one batch group.
    pub fn detect(
        &self,
        decode_candidates: &[SessionId],
        session_states: &HashMap<SessionId, BatchSessionInfo>,
    ) -> Vec<BatchGroup> {
        let mut eligible: Vec<SessionId> = decode_candidates
            .iter()
            .copied()
            .filter(|id| {
                session_states
                    .get(id)
                    .map(|info| info.is_decoding && info.has_current_token)
                    .unwrap_or(false)
            })
            .collect();

        if eligible.len() < self.min_batch_size {
            // Not enough sessions to batch — return each as its own group
            // (caller will decode serially).
            return eligible
                .into_iter()
                .map(|id| BatchGroup::new(vec![id]))
                .collect();
        }

        // All eligible sessions form one batch group.
        vec![BatchGroup::new(eligible)]
    }

    /// Split a batch group into sub-groups that respect a max batch size.
    ///
    /// Useful when thermal policy limits how many sessions can be active
    /// in one scheduling tick.
    pub fn split_by_max_batch(
        group: BatchGroup,
        max_per_batch: usize,
    ) -> Vec<BatchGroup> {
        if max_per_batch == 0 || group.session_ids.is_empty() {
            return Vec::new();
        }
        group
            .session_ids
            .chunks(max_per_batch)
            .map(|chunk| BatchGroup::new(chunk.to_vec()))
            .collect()
    }
}

// BatchSessionInfo 

/// Lightweight snapshot of a session's decode-readiness.
///
/// The Engine builds this from its internal state before calling the
/// BatchDetector.
#[derive(Debug, Clone, Copy)]
pub struct BatchSessionInfo {
    /// Session is in the Decoding state (not Queued/Prefill/Suspended/Terminal).
    pub is_decoding: bool,
    /// Session has a current token to feed into the next decode step.
    pub has_current_token: bool,
    /// Number of KV cache tokens (used for priority ordering in Throughput mode).
    pub token_count: usize,
}

impl BatchSessionInfo {
    pub fn new(is_decoding: bool, has_current_token: bool, token_count: usize) -> Self {
        Self {
            is_decoding,
            has_current_token,
            token_count,
        }
    }
}

// BatchedDecodeResult 

/// The result of a batched decode step — one token per session in the batch.
#[derive(Debug, Clone)]
pub struct BatchedDecodeResult {
    /// Per-session outputs: (SessionId, next_token).
    pub tokens: Vec<(SessionId, u32)>,
    /// Number of sessions that hit a stop token this decode step.
    pub stop_count: usize,
}

impl BatchedDecodeResult {
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            stop_count: 0,
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_detector_groups_eligible_sessions() {
        let detector = BatchDetector::new(); // min_batch_size = 2
        let candidates = vec![1, 2, 3];
        let mut states = HashMap::new();
        states.insert(1, BatchSessionInfo::new(true, true, 128));
        states.insert(2, BatchSessionInfo::new(true, true, 56));
        states.insert(3, BatchSessionInfo::new(true, true, 200));

        let groups = detector.detect(&candidates, &states);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].batch_size(), 3);
        assert_eq!(groups[0].session_ids, vec![1, 2, 3]);
    }

    #[test]
    fn batch_detector_filters_non_decoding() {
        let detector = BatchDetector::new();
        let candidates = vec![1, 2, 3];
        let mut states = HashMap::new();
        states.insert(1, BatchSessionInfo::new(true, true, 100));
        states.insert(2, BatchSessionInfo::new(false, false, 0)); // suspended
        states.insert(3, BatchSessionInfo::new(true, true, 50));

        let groups = detector.detect(&candidates, &states);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].session_ids, vec![1, 3]); // session 2 filtered out
    }

    #[test]
    fn batch_detector_below_min_returns_singletons() {
        let detector = BatchDetector::new().with_min_batch_size(3);
        let candidates = vec![1, 2];
        let mut states = HashMap::new();
        states.insert(1, BatchSessionInfo::new(true, true, 100));
        states.insert(2, BatchSessionInfo::new(true, true, 50));

        let groups = detector.detect(&candidates, &states);
        // Only 2 eligible, below min_batch_size=3 → each gets its own group.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].session_ids, vec![1]);
        assert_eq!(groups[1].session_ids, vec![2]);
    }

    #[test]
    fn batch_detector_filters_no_token() {
        let detector = BatchDetector::new();
        let candidates = vec![1, 2];
        let mut states = HashMap::new();
        states.insert(1, BatchSessionInfo::new(true, true, 100));
        states.insert(2, BatchSessionInfo::new(true, false, 0)); // no current token

        let groups = detector.detect(&candidates, &states);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].session_ids, vec![1]);
    }

    #[test]
    fn split_by_max_batch() {
        let group = BatchGroup::new(vec![1, 2, 3, 4, 5]);
        let subgroups = BatchDetector::split_by_max_batch(group, 2);
        assert_eq!(subgroups.len(), 3);
        assert_eq!(subgroups[0].session_ids, vec![1, 2]);
        assert_eq!(subgroups[1].session_ids, vec![3, 4]);
        assert_eq!(subgroups[2].session_ids, vec![5]);
    }

    #[test]
    fn split_empty_group() {
        let group = BatchGroup::new(vec![]);
        let subgroups = BatchDetector::split_by_max_batch(group, 2);
        assert!(subgroups.is_empty());
    }

    #[test]
    fn batch_group_properties() {
        let group = BatchGroup::new(vec![10, 20, 30]);
        assert_eq!(group.batch_size(), 3);
        assert_eq!(group.tokens_per_session, 1);
        assert!(!group.is_empty());

        let empty = BatchGroup::new(vec![]);
        assert!(empty.is_empty());
    }
}
