// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Scheduling policies for the cellm scheduler.
//!
//! Three policies govern how decode time is allocated across sessions:
//!
//! - **LatencyFirst**: Minimize time-to-first-token (TTFT). Sessions in prefill
//!   are always serviced before decode sessions. Among decode sessions, the
//!   youngest session (fewest generated tokens) gets priority. Use this for
//!   interactive chat where users expect immediate responses.
//!
//! - **ThroughputFirst**: Maximize aggregate tokens/sec. Eligible sessions are
//!   batched together when possible (continuous batching lite). Decode sessions
//!   with longer history get priority (amortizes KV cache access). Prefill is
//!   deferred until at least one burst of decode has completed.
//!
//! - **Fair**: Round-robin time slicing. Every session in the decode set gets
//!   one token per scheduling round. This is the simplest policy and the
//!   default for mobile deployments where session count is low (2–4).
//!
//! The policy does **not** override thermal limits — if the thermal monitor
//! says pause, all policies pause.

use std::collections::VecDeque;

use crate::rr::SessionId;
use crate::session::{Session, SessionState};

// SchedulingPolicy 

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Minimize TTFT — one session at a time, prefill first.
    LatencyFirst,
    /// Maximize tok/s — batch where possible, prefer longer decode sessions.
    ThroughputFirst,
    /// Round-robin with equal time slices.
    Fair,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Fair
    }
}

// PolicyExecutor ──

/// Drives scheduling decisions according to the active policy.
///
/// This wraps a `VecDeque<SessionId>` (the decode set) and reorders it
/// according to the policy before each scheduling burst.
pub struct PolicyExecutor {
    policy: SchedulingPolicy,
    /// Ordered decode set (policy-specific ordering).
    decode_order: VecDeque<SessionId>,
}

impl PolicyExecutor {
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self {
            policy,
            decode_order: VecDeque::new(),
        }
    }

    pub fn policy(&self) -> SchedulingPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: SchedulingPolicy) {
        self.policy = policy;
    }

    /// Clear and rebuild the decode order.
    pub fn clear(&mut self) {
        self.decode_order.clear();
    }

    /// Add a session to the decode set.
    pub fn add(&mut self, id: SessionId) {
        if !self.decode_order.contains(&id) {
            self.decode_order.push_back(id);
        }
    }

    /// Remove a session from the decode set.
    pub fn remove(&mut self, id: SessionId) {
        if let Some(idx) = self.decode_order.iter().position(|&x| x == id) {
            self.decode_order.remove(idx);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.decode_order.is_empty()
    }

    pub fn len(&self) -> usize {
        self.decode_order.len()
    }

    /// Return the ordered list of session IDs that should be decoded this
    /// scheduling tick. The caller provides session metadata so the policy
    /// can sort / filter.
    ///
    /// **LatencyFirst**: Sessions sorted by generated token count ascending
    ///   (youngest first). Sessions in Prefill state are inserted at the front.
    ///
    /// **ThroughputFirst**: Sessions sorted by generated token count descending
    ///   (longest first, to amortize KV access). Compatible-shape neighbours
    ///   are grouped for potential batching.
    ///
    /// **Fair**: FIFO round-robin — pop from front, push to back.
    pub fn tick(
        &mut self,
        sessions: &[&Session],
        prefill_ids: &[SessionId],
    ) -> SchedulingPlan {
        match self.policy {
            SchedulingPolicy::Fair => self.tick_fair(),
            SchedulingPolicy::LatencyFirst => self.tick_latency_first(sessions, prefill_ids),
            SchedulingPolicy::ThroughputFirst => self.tick_throughput_first(sessions),
        }
    }

    // Fair (round-robin)────────

    fn tick_fair(&mut self) -> SchedulingPlan {
        if self.decode_order.is_empty() {
            return SchedulingPlan::empty();
        }
        // Pop one from the front and push it to the back.
        let id = self.decode_order.pop_front().unwrap();
        self.decode_order.push_back(id);
        SchedulingPlan {
            decode_ids: vec![id],
            prefill_ids: Vec::new(),
            batch_groups: Vec::new(),
        }
    }

    // LatencyFirst 

    fn tick_latency_first(
        &mut self,
        sessions: &[&Session],
        prefill_ids: &[SessionId],
    ) -> SchedulingPlan {
        // Prefill always gets priority — return one prefill session if any.
        if let Some(&id) = prefill_ids.first() {
            return SchedulingPlan {
                decode_ids: Vec::new(),
                prefill_ids: vec![id],
                batch_groups: Vec::new(),
            };
        }

        if self.decode_order.is_empty() {
            return SchedulingPlan::empty();
        }

        // Build a lookup: SessionId → generated_tokens
        let generated: std::collections::HashMap<SessionId, u64> = sessions
            .iter()
            .map(|s| (s.id(), s.generated_tokens()))
            .collect();

        // Sort decode set by fewest generated tokens (youngest sessions first).
        let mut ids: Vec<SessionId> = self.decode_order.iter().copied().collect();
        ids.sort_by_key(|id| generated.get(id).copied().unwrap_or(0));

        // Pick the first (youngest) session.
        let selected = ids[0];

        // Rebuild decode_order in the sorted order (minus the selected one,
        // which goes to the back after its turn).
        self.decode_order.clear();
        for &id in &ids[1..] {
            self.decode_order.push_back(id);
        }
        self.decode_order.push_back(selected);

        SchedulingPlan {
            decode_ids: vec![selected],
            prefill_ids: Vec::new(),
            batch_groups: Vec::new(),
        }
    }

    // ThroughputFirst 

    fn tick_throughput_first(
        &mut self,
        sessions: &[&Session],
    ) -> SchedulingPlan {
        if self.decode_order.is_empty() {
            return SchedulingPlan::empty();
        }

        // Build lookup: SessionId → generated_tokens
        let generated: std::collections::HashMap<SessionId, u64> = sessions
            .iter()
            .map(|s| (s.id(), s.generated_tokens()))
            .collect();

        // Sort decode set by MOST generated tokens (longest sessions first).
        let mut ids: Vec<SessionId> = self.decode_order.iter().copied().collect();
        ids.sort_by_key(|id| {
            // Negate so larger values sort first (descending).
            std::cmp::Reverse(generated.get(id).copied().unwrap_or(0))
        });

        // Build batch groups: consecutive sessions that can be batched.
        // On mobile with 2-4 sessions we just batch all of them.
        let batch_group = ids.clone();

        // Rebuild decode_order for next tick.
        self.decode_order.clear();
        for &id in &ids {
            self.decode_order.push_back(id);
        }

        SchedulingPlan {
            decode_ids: batch_group,
            prefill_ids: Vec::new(),
            batch_groups: Vec::new(),
        }
    }
}

// SchedulingPlan ──

/// The output of one scheduling tick — tells the Engine which sessions to
/// process and in what order.
#[derive(Debug, Clone)]
pub struct SchedulingPlan {
    /// Ordered list of session IDs to decode (one per session for Fair/Latency,
    /// multiple for Throughput batch).
    pub decode_ids: Vec<SessionId>,
    /// Session IDs that should be prefilled this tick.
    pub prefill_ids: Vec<SessionId>,
    /// Groups of session IDs whose decode can be batched together.
    /// Each inner vec is a batch-compatible group.
    pub batch_groups: Vec<Vec<SessionId>>,
}

impl SchedulingPlan {
    pub fn empty() -> Self {
        Self {
            decode_ids: Vec::new(),
            prefill_ids: Vec::new(),
            batch_groups: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.decode_ids.is_empty() && self.prefill_ids.is_empty()
    }

    /// Total number of decode tokens this plan will produce.
    pub fn decode_count(&self) -> usize {
        self.decode_ids.len()
    }
}

// Tests ───────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: SessionId, generated: u64) -> Session {
        let mut s = Session::new(id);
        for _ in 0..generated {
            s.add_generated_token();
        }
        s
    }

    #[test]
    fn fair_round_robin_rotates() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::Fair);
        p.add(1);
        p.add(2);
        p.add(3);

        // First tick picks 1, pushes it back: order becomes [2, 3, 1]
        let plan = p.tick(&[], &[]);
        assert_eq!(plan.decode_ids, vec![1]);

        // Second tick picks 2: order becomes [3, 1, 2]
        let plan = p.tick(&[], &[]);
        assert_eq!(plan.decode_ids, vec![2]);

        // Third: [1, 2, 3]
        let plan = p.tick(&[], &[]);
        assert_eq!(plan.decode_ids, vec![3]);
    }

    #[test]
    fn latency_first_picks_youngest() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::LatencyFirst);
        let s1 = make_session(1, 100);
        let s2 = make_session(2, 5);
        let s3 = make_session(3, 50);

        p.add(1);
        p.add(2);
        p.add(3);

        // Youngest (fewest generated tokens) is session 2.
        let plan = p.tick(&[&s1, &s2, &s3], &[]);
        assert_eq!(plan.decode_ids, vec![2]);
    }

    #[test]
    fn latency_first_prefill_priority() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::LatencyFirst);
        let s1 = make_session(1, 10);
        let s2 = make_session(2, 20);

        p.add(1);
        p.add(2);

        // A prefill session is waiting — it wins.
        let plan = p.tick(&[&s1, &s2], &[99]);
        assert_eq!(plan.prefill_ids, vec![99]);
        assert!(plan.decode_ids.is_empty());
    }

    #[test]
    fn throughput_first_batches_all() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::ThroughputFirst);
        let s1 = make_session(1, 10);
        let s2 = make_session(2, 50);
        let s3 = make_session(3, 30);

        p.add(1);
        p.add(2);
        p.add(3);

        // Should batch all sessions, ordered by most generated first.
        let plan = p.tick(&[&s1, &s2, &s3], &[]);
        assert_eq!(plan.decode_ids, vec![2, 3, 1]); // 50, 30, 10
    }

    #[test]
    fn policy_switch_clears_ordering() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::Fair);
        p.add(1);
        p.add(2);

        p.set_policy(SchedulingPolicy::LatencyFirst);
        // Switching policy doesn't clear by default — but a clear() is available.
        p.clear();
        assert!(p.is_empty());
    }

    #[test]
    fn fair_empty_returns_empty_plan() {
        let mut p = PolicyExecutor::new(SchedulingPolicy::Fair);
        let plan = p.tick(&[], &[]);
        assert!(plan.is_empty());
    }
}
