// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Batched forward pass for continuous batching.
//!
//! When multiple sessions are in the decode phase, their weight-bound matmuls
//! (QKV projection, output projection, MLP) use the same model weights. This
//! module batches those matmuls by stacking session hidden states into a
//! [batch_size, hidden_dim] matrix and calling sgemm once per operation instead
//! of batch_size times.
//!
//! Per-session operations (RoPE at different positions, attention over
//! different KV histories, KV cache writes) remain unbatched.
//!
//! Data flow:
//!
//! ```text
//! tokens[B] -> embed -> X[B, hidden]
//! for each layer:
//!     X_norm = rms_norm_batched(X, weight)
//!     QKV = sgemm(X_norm, W_qkv_concat)           / batched
//!     split Q/K/V per session
//!     for each session:
//!         RoPE(q, k, pos)                          / per-session
//!         write_kv(k, v, cache, pos)               / per-session
//!         attn_out = attention(q, cache)           / per-session
//!     X += sgemm(attn_out, W_o)                    / batched
//!     X += mlp_batched(rms_norm(X))                / batched
//! logits = sgemm(X_norm, W_lm_head)                / batched
//! split logits per session -> sample
//! ```

use bytemuck::cast_slice;
use cellm_cache::{KVCache, PageTable};
use cellm_core::CoreError;
use cellm_kernels::cpu_kernels::rms_norm_f32;
use half::f16;

use crate::{CellmFile, ModelConfig};

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        Order: i32,
        TransA: i32,
        TransB: i32,
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: f32,
        C: *mut f32,
        ldc: i32,
    );
}

/// A single session's decode state for the batched forward pass.
pub struct BatchedDecodeHandle<'a> {
    pub session_id: u64,
    pub token: u32,
    pub pos: usize,
    pub page_table: &'a mut PageTable,
    pub kv_cache: &'a mut KVCache,
}

/// Output from one batched decode step, returned per session.
pub struct BatchedStepOutput {
    pub session_id: u64,
    pub candidates: Vec<(u32, f32)>,
    pub is_stop: bool,
}

/// Executes a batched forward pass for multiple sessions sharing one model.
///
/// Created by the Engine in ThroughputFirst mode when multiple sessions are
/// ready for decode. Borrows model weights from a CellmFile and processes all
/// sessions in one forward call using batched sgemm.
pub struct BatchedForward<'a> {
    file: &'a CellmFile,
    cfg: &'a ModelConfig,
    eos_token_id: Option<u32>,
    max_layers: usize,
}

impl<'a> BatchedForward<'a> {
    pub fn new(
        file: &'a CellmFile,
        cfg: &'a ModelConfig,
        eos_token_id: Option<u32>,
        max_layers: usize,
    ) -> Self {
        Self {
            file,
            cfg,
            eos_token_id,
            max_layers,
        }
    }

    /// Run one batched decode step: produce one next token for each session.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn step_topk_batched(
        &mut self,
        handles: &mut [BatchedDecodeHandle<'_>],
        top_k: usize,
    ) -> Result<Vec<BatchedStepOutput>, CoreError> {
        let batch_size = handles.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let hidden = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let n_kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = self.cfg.intermediate_size;

        // Embed all tokens into [B, hidden] matrix.
        let mut x_batch = vec![0.0f32; batch_size * hidden];
        for (s, h) in handles.iter().enumerate() {
            let start = s * hidden;
            embed_token_from_file(self.file, h.token, hidden, &mut x_batch[start..start + hidden])?;
        }

        // Precompute layer tensor names.
        let layer_names: Vec<LayerNames> = (0..self.max_layers)
            .map(|l| LayerNames {
                attn_norm: format!("model.layers.{l}.input_layernorm.weight"),
                q_proj: format!("model.layers.{l}.self_attn.q_proj.weight"),
                k_proj: format!("model.layers.{l}.self_attn.k_proj.weight"),
                v_proj: format!("model.layers.{l}.self_attn.v_proj.weight"),
                o_proj: format!("model.layers.{l}.self_attn.o_proj.weight"),
                ffn_norm: format!("model.layers.{l}.post_attention_layernorm.weight"),
                gate_proj: format!("model.layers.{l}.mlp.gate_proj.weight"),
                up_proj: format!("model.layers.{l}.mlp.up_proj.weight"),
                down_proj: format!("model.layers.{l}.mlp.down_proj.weight"),
            })
            .collect();

        // Per-batch scratch buffers.
        let mut x_norm_batch = vec![0.0f32; batch_size * hidden];
        let total_qkv = hidden + kv_dim + kv_dim;
        let mut qkv_batch = vec![0.0f32; batch_size * total_qkv];
        let mut attn_out_batch = vec![0.0f32; batch_size * hidden];
        let mut attn_proj_batch = vec![0.0f32; batch_size * hidden];
        let mut mlp_in_batch = vec![0.0f32; batch_size * hidden];
        let mut gate_batch = vec![0.0f32; batch_size * inter];
        let mut up_batch = vec![0.0f32; batch_size * inter];

        // Per-session scratch (reused across layers).
        let mut q_single = vec![0.0f32; hidden];
        let mut k_single = vec![0.0f32; kv_dim];
        let mut v_single = vec![0.0f32; kv_dim];
        let mut attn_single = vec![0.0f32; hidden];
        let mut gather_bases: Vec<usize> = Vec::new();
        let mut norm_w_single = vec![0.0f32; hidden];

        for layer in 0..self.max_layers {
            let ln = &layer_names[layer];

            // Batched RMSNorm (attention input norm).
            load_f32_from_file(self.file, &ln.attn_norm, &mut norm_w_single)?;
            for s in 0..batch_size {
                let base = s * hidden;
                rms_norm_f32(
                    &x_batch[base..base + hidden],
                    &norm_w_single,
                    self.cfg.rms_norm_eps,
                    &mut x_norm_batch[base..base + hidden],
                );
            }

            // Batched QKV projection: one sgemm instead of three.
            let qkv_weight = load_concat_qkv_from_file(
                self.file, &ln.q_proj, hidden, &ln.k_proj, kv_dim, &ln.v_proj, kv_dim, hidden,
            )?;
            unsafe {
                cblas_sgemm(
                    101, 111, 112,
                    batch_size as i32, total_qkv as i32, hidden as i32,
                    1.0f32,
                    x_norm_batch.as_ptr(), hidden as i32,
                    qkv_weight.as_ptr(), total_qkv as i32,
                    0.0f32,
                    qkv_batch.as_mut_ptr(), total_qkv as i32,
                );
            }

            // Per-session: split QKV, RoPE, write KV, attention.
            for s in 0..batch_size {
                let q_start = s * total_qkv;
                let k_start = q_start + hidden;
                let v_start = k_start + kv_dim;

                q_single.copy_from_slice(&qkv_batch[q_start..q_start + hidden]);
                k_single.copy_from_slice(&qkv_batch[k_start..k_start + kv_dim]);
                v_single.copy_from_slice(&qkv_batch[v_start..v_start + kv_dim]);

                let h = &mut handles[s];
                let pos = h.pos;

                if pos == h.page_table.token_count() {
                    h.page_table.append_token(h.kv_cache.allocator_mut()).map_err(|e| {
                        CoreError::Backend(format!("batched: append_token: {e}"))
                    })?;
                }

                let block_id = h.page_table.block_for_token(pos).map_err(|e| {
                    CoreError::Backend(format!("batched: block_for_token: {e}"))
                })?;
                let token_off = h.page_table.offset_in_block(pos).map_err(|e| {
                    CoreError::Backend(format!("batched: offset_in_block: {e}"))
                })?;

                // RoPE.
                cellm_kernels::cpu_kernels::rope_non_interleaved_inplace_f32(
                    &mut q_single, n_heads, head_dim, head_dim, pos, self.cfg.rope_theta,
                );
                cellm_kernels::cpu_kernels::rope_non_interleaved_inplace_f32(
                    &mut k_single, n_kv_heads, head_dim, head_dim, pos, self.cfg.rope_theta,
                );

                // Write K/V.
                {
                    let mut cv = h.kv_cache.view_mut();
                    cv.write_token(block_id, layer, token_off, &k_single, &v_single)?;
                }

                // Gather and attend.
                let seq = h.page_table.token_count();
                let cr = h.kv_cache.view();
                gather_bases.clear();
                gather_bases.reserve(seq);
                for tpos in 0..seq {
                    let b = h.page_table.block_for_token(tpos).map_err(|e| {
                        CoreError::Backend(format!("batched: block_for_token(gather): {e}"))
                    })?;
                    let o = h.page_table.offset_in_block(tpos).map_err(|e| {
                        CoreError::Backend(format!("batched: offset_in_block(gather): {e}"))
                    })?;
                    gather_bases.push(cr.layout.token_base_elem(b, layer, o)?);
                }
                cr.attention_single_token_gqa_from_bases(
                    &gather_bases, &q_single,
                    n_heads, n_kv_heads, head_dim,
                    None, None,
                    &mut attn_single,
                )?;

                let ao_start = s * hidden;
                attn_out_batch[ao_start..ao_start + hidden].copy_from_slice(&attn_single);
            }

            // Batched output projection.
            let o_weight = load_f32_vec_from_file(self.file, &ln.o_proj)?;
            unsafe {
                cblas_sgemm(
                    101, 111, 112,
                    batch_size as i32, hidden as i32, hidden as i32,
                    1.0f32,
                    attn_out_batch.as_ptr(), hidden as i32,
                    o_weight.as_ptr(), hidden as i32,
                    0.0f32,
                    attn_proj_batch.as_mut_ptr(), hidden as i32,
                );
            }

            // Batched residual add.
            for i in 0..(batch_size * hidden) {
                x_batch[i] += attn_proj_batch[i];
            }

            // Batched FFN norm.
            load_f32_from_file(self.file, &ln.ffn_norm, &mut norm_w_single)?;
            for s in 0..batch_size {
                let base = s * hidden;
                rms_norm_f32(
                    &x_batch[base..base + hidden],
                    &norm_w_single,
                    self.cfg.rms_norm_eps,
                    &mut mlp_in_batch[base..base + hidden],
                );
            }

            // Batched MLP: gate, up, SiLU, down.
            let gate_w = load_f32_vec_from_file(self.file, &ln.gate_proj)?;
            unsafe {
                cblas_sgemm(
                    101, 111, 112,
                    batch_size as i32, inter as i32, hidden as i32,
                    1.0f32, mlp_in_batch.as_ptr(), hidden as i32,
                    gate_w.as_ptr(), inter as i32,
                    0.0f32, gate_batch.as_mut_ptr(), inter as i32,
                );
            }

            let up_w = load_f32_vec_from_file(self.file, &ln.up_proj)?;
            unsafe {
                cblas_sgemm(
                    101, 111, 112,
                    batch_size as i32, inter as i32, hidden as i32,
                    1.0f32, mlp_in_batch.as_ptr(), hidden as i32,
                    up_w.as_ptr(), inter as i32,
                    0.0f32, up_batch.as_mut_ptr(), inter as i32,
                );
            }

            // SiLU(gate) * up in-place into gate_batch.
            for i in 0..(batch_size * inter) {
                let g = gate_batch[i];
                let sig = 1.0f32 / (1.0f32 + (-g).exp());
                gate_batch[i] = g * sig * up_batch[i];
            }

            let down_w = load_f32_vec_from_file(self.file, &ln.down_proj)?;
            unsafe {
                cblas_sgemm(
                    101, 111, 112,
                    batch_size as i32, hidden as i32, inter as i32,
                    1.0f32, gate_batch.as_ptr(), inter as i32,
                    down_w.as_ptr(), hidden as i32,
                    1.0f32, x_batch.as_mut_ptr(), hidden as i32,
                );
            }
        }

        // Final RMSNorm.
        let final_norm_name = "model.norm.weight";
        load_f32_from_file(self.file, final_norm_name, &mut norm_w_single)?;
        for s in 0..batch_size {
            let base = s * hidden;
            rms_norm_f32(
                &x_batch[base..base + hidden],
                &norm_w_single,
                self.cfg.rms_norm_eps,
                &mut x_norm_batch[base..base + hidden],
            );
        }

        // Batched logit projection.
        let vocab = self.cfg.vocab_size;
        let mut logits_batch = vec![0.0f32; batch_size * vocab];
        let lm_head_key = "lm_head.weight";
        let embed_key = "model.embed_tokens.weight";
        let head_weight = load_lm_head_or_embed(self.file, lm_head_key, embed_key, vocab, hidden)?;
        unsafe {
            cblas_sgemm(
                101, 111, 112,
                batch_size as i32, vocab as i32, hidden as i32,
                1.0f32, x_norm_batch.as_ptr(), hidden as i32,
                head_weight.as_ptr(), vocab as i32,
                0.0f32, logits_batch.as_mut_ptr(), vocab as i32,
            );
        }

        // Per-session top-k.
        let mut results = Vec::with_capacity(batch_size);
        for s in 0..batch_size {
            let start = s * vocab;
            let logits = &logits_batch[start..start + vocab];
            let candidates = topk_from_slice(logits, top_k);
            let is_stop = candidates
                .first()
                .map(|&(id, _)| Some(id) == self.eos_token_id)
                .unwrap_or(false);
            results.push(BatchedStepOutput {
                session_id: handles[s].session_id,
                candidates,
                is_stop,
            });
        }

        Ok(results)
    }
}

// Free functions (use public CellmFile API)─

fn load_f32_from_file(
    file: &CellmFile,
    name: &str,
    out: &mut Vec<f32>,
) -> Result<(), CoreError> {
    let data = file.tensor_bytes(name).map_err(|_| {
        CoreError::Backend(format!("batched: weight not found: {name}"))
    })?;
    let meta = file.tensor_index(name).ok_or_else(|| {
        CoreError::Backend(format!("batched: tensor meta not found: {name}"))
    })?;
    cast_to_f32_into(data, meta.dtype.as_str(), out)
}

fn load_f32_vec_from_file(file: &CellmFile, name: &str) -> Result<Vec<f32>, CoreError> {
    let data = file.tensor_bytes(name).map_err(|_| {
        CoreError::Backend(format!("batched: weight not found: {name}"))
    })?;
    let meta = file.tensor_index(name).ok_or_else(|| {
        CoreError::Backend(format!("batched: tensor meta not found: {name}"))
    })?;
    cast_to_f32(data, meta.dtype.as_str())
}

fn cast_to_f32_into(data: &[u8], dtype: &str, out: &mut Vec<f32>) -> Result<(), CoreError> {
    out.clear();
    match dtype {
        "f16" => {
            let f16s: &[f16] = cast_slice(data);
            out.extend(f16s.iter().map(|&v| v.to_f32()));
        }
        "f32" => {
            let f32s: &[f32] = cast_slice(data);
            out.extend_from_slice(f32s);
        }
        other => {
            return Err(CoreError::Backend(format!(
                "batched: unsupported dtype: {other}"
            )));
        }
    }
    Ok(())
}

fn cast_to_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, CoreError> {
    match dtype {
        "f16" => {
            let f16s: &[f16] = cast_slice(data);
            Ok(f16s.iter().map(|&v| v.to_f32()).collect())
        }
        "f32" => {
                    let f32s: &[f32] = cast_slice(data);
                    Ok(f32s.to_vec())
                }
        other => Err(CoreError::Backend(format!(
            "batched: unsupported dtype: {other}"
        ))),
    }
}

fn embed_token_from_file(
    file: &CellmFile,
    token: u32,
    hidden: usize,
    out: &mut [f32],
) -> Result<(), CoreError> {
    let embed_key = "model.embed_tokens.weight";
    let meta = file.tensor_index(embed_key).ok_or_else(|| {
        CoreError::Backend("batched: embed_tokens not found".into())
    })?;
    let data = file.tensor_bytes(embed_key).map_err(|_| {
        CoreError::Backend("batched: embed_tokens bytes not found".into())
    })?;

    let vocab_size = meta.shape[0];
    if (token as usize) >= vocab_size {
        return Err(CoreError::Backend(format!(
            "batched: token {token} out of vocab range ({vocab_size})"
        )));
    }

    match meta.dtype.as_str() {
        "f16" => {
            let f16s: &[f16] = cast_slice(data);
            let start = (token as usize) * hidden;
            for i in 0..hidden {
                out[i] = f16s[start + i].to_f32();
            }
        }
        "f32" => {
            let f32s: &[f32] = cast_slice(data);
            let start = (token as usize) * hidden;
            out.copy_from_slice(&f32s[start..start + hidden]);
        }
        other => {
            return Err(CoreError::Backend(format!(
                "batched: unsupported embed dtype: {other}"
            )));
        }
    }
    Ok(())
}

fn load_concat_qkv_from_file(
    file: &CellmFile,
    q_name: &str, q_out: usize,
    k_name: &str, k_out: usize,
    v_name: &str, v_out: usize,
    in_dim: usize,
) -> Result<Vec<f32>, CoreError> {
    let q_w = load_f32_vec_from_file(file, q_name)?;
    let k_w = load_f32_vec_from_file(file, k_name)?;
    let v_w = load_f32_vec_from_file(file, v_name)?;

    let total_out = q_out + k_out + v_out;
    let mut concat = vec![0.0f32; total_out * in_dim];

    for row in 0..q_out {
        let src = row * in_dim;
        concat[src..src + in_dim].copy_from_slice(&q_w[src..src + in_dim]);
    }
    for row in 0..k_out {
        let src = row * in_dim;
        let dst = (q_out + row) * in_dim;
        concat[dst..dst + in_dim].copy_from_slice(&k_w[src..src + in_dim]);
    }
    for row in 0..v_out {
        let src = row * in_dim;
        let dst = (q_out + k_out + row) * in_dim;
        concat[dst..dst + in_dim].copy_from_slice(&v_w[src..src + in_dim]);
    }

    Ok(concat)
}

fn load_lm_head_or_embed(
    file: &CellmFile,
    lm_head_key: &str,
    embed_key: &str,
    vocab: usize,
    hidden: usize,
) -> Result<Vec<f32>, CoreError> {
    if let Ok(data) = file.tensor_bytes(lm_head_key) {
        if let Some(meta) = file.tensor_index(lm_head_key) {
            let v = cast_to_f32(data, meta.dtype.as_str())?;
            if v.len() == vocab * hidden {
                return Ok(v);
            }
        }
    }
    let data = file.tensor_bytes(embed_key).map_err(|_| {
        CoreError::Backend("batched: lm_head and embed_tokens not found".into())
    })?;
    let meta = file.tensor_index(embed_key).ok_or_else(|| {
        CoreError::Backend("batched: embed_tokens meta not found".into())
    })?;
    cast_to_f32(data, meta.dtype.as_str())
}

struct LayerNames {
    attn_norm: String,
    q_proj: String,
    k_proj: String,
    v_proj: String,
    o_proj: String,
    ffn_norm: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

fn topk_from_slice(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = logits.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
}

// Tests 

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_basic() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let top = topk_from_slice(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 3);
        assert_eq!(top[1].0, 1);
        assert_eq!(top[2].0, 2);
    }

    #[test]
    fn topk_empty() {
        let top = topk_from_slice(&[], 5);
        assert!(top.is_empty());
    }

    #[test]
    fn topk_k_larger_than_len() {
        let logits = vec![0.5f32, 0.8];
        let top = topk_from_slice(&logits, 10);
        assert_eq!(top.len(), 2);
    }
}
