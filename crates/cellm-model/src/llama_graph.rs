// Author: Jeffrey Asante (https://jeffasante.github.io/)
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::HashMap;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_core::CoreError;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_kernels::metal::MetalOps;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use cellm_cache::{KVCache, PageTable};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::ffi::c_void;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct LlamaGraphState {
    pub device: Device,
    pub queue: CommandQueue,

    // Core Ops
    pub ops: MetalOps,

    // Pre-allocated Model Weights
    pub weights: HashMap<String, Buffer>,

    // Persistent Activations (Zero-Copy sequence dispatch)
    pub x_buf: Buffer,
    pub x_norm_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub mlp_in_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,
    pub bases_buf: Option<Buffer>,
    pub bases_stride: usize,       // allocated tokens-per-layer (power of two, >= seq)
    pub bases_last_session: u64,   // session that last wrote to this buffer
    pub bases_last_seq: usize,     // token_count when buffer was last updated
    /// When false, use rotate-half RoPE (e.g. SmolLM2); when true, use adjacent-pair RoPE (standard Llama).
    pub rope_interleaved: bool,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl LlamaGraphState {
    pub fn set_rope_interleaved(&mut self, v: bool) {
        self.rope_interleaved = v;
    }

    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        vocab_size: usize,
        intermediate_size: usize,
        rope_interleaved: bool,
    ) -> Result<Self, CoreError> {
        let ops = MetalOps::create().map_err(|e| CoreError::Backend(format!("{:?}", e)))?;
        let head_dim = hidden_size / num_heads;

        let device = ops.device.clone();
        let queue = ops.queue.clone();

        let make_buf = |len_f32: usize| {
            device.new_buffer(
                (len_f32 * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Activations are mapped specifically for f32 operations between kernel execution passes
        Ok(Self {
            device: device.clone(),
            queue,
            ops,
            weights: HashMap::new(),
            x_buf: make_buf(hidden_size),
            x_norm_buf: make_buf(hidden_size),
            q_buf: make_buf(hidden_size),
            k_buf: make_buf(num_kv_heads * head_dim),
            v_buf: make_buf(num_kv_heads * head_dim),
            attn_out_buf: make_buf(hidden_size),
            mlp_in_buf: make_buf(hidden_size),
            gate_buf: make_buf(intermediate_size),
            up_buf: make_buf(intermediate_size),
            down_buf: make_buf(hidden_size),
            logits_buf: make_buf(vocab_size),
            bases_buf: None,
            bases_stride: 0,
            bases_last_session: 0,
            bases_last_seq: 0,
            rope_interleaved,
        })
    }

    pub fn reserve_sequence_capacity(&mut self, max_seq: usize, num_layers: usize) {
        let new_stride = max_seq.next_power_of_two().max(64);
        if self.bases_stride < new_stride || self.bases_buf.is_none() {
            self.bases_stride = new_stride;
            self.bases_buf = Some(self.device.new_buffer(
                (new_stride * num_layers * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.bases_last_seq = 0;
        }
    }

    /// Batched prefill: process all tokens in a single command buffer.
    /// This eliminates N-1 command-buffer creations and GPU syncs that
    /// the per-token `step_fused` path incurs.
    ///
    /// IMPORTANT: We upload ALL token embeddings into a single GPU buffer
    /// upfront, then use Metal blit encoders between per-token compute
    /// passes to copy each token's data into `x_buf`. This avoids the
    /// CPU-GPU synchronization hazard of writing `x_buf` from the CPU
    /// while the command buffer is being encoded (the GPU would only see
    /// the last write).
    pub fn prefill_fused(
        &mut self,
        x_all: &[f32],
        cfg: &crate::ModelConfig,
        prefix: &str,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        start_pos: usize,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        let _kv_dim = n_kv_heads * head_dim;
        let num_layers = cfg.num_hidden_layers;
        let num_tokens = x_all.len() / hidden;

        let seq = page_table.token_count();
        let session_id = page_table.session_id();

        // ---- 1. Build bases buffer for all tokens ----
        let need_resize = self.bases_stride < seq;
        if need_resize || self.bases_buf.is_none() {
            let new_stride = seq.next_power_of_two().max(64);
            self.bases_stride = new_stride;
            self.bases_buf = Some(self.device.new_buffer(
                (new_stride * num_layers * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.bases_last_seq = 0;
        }

        let layout = kv_cache.layout();
        let elems_per_layer = layout.elems_per_block_per_layer() as u32;
        let stride = self.bases_stride;
        let bases_ptr = self.bases_buf.as_ref().unwrap().contents() as *mut u32;

        let incremental = !need_resize
            && self.bases_last_session == session_id
            && self.bases_last_seq + num_tokens == seq;

        if incremental {
            for t in (seq - num_tokens)..seq {
                let pos = t;
                let b = page_table.block_for_token(pos)
                    .map_err(|e| CoreError::Backend(format!("prefill bases: {e}")))?;
                let o = page_table.offset_in_block(pos)
                    .map_err(|e| CoreError::Backend(format!("prefill bases: {e}")))?;
                let base_l0 = layout.token_base_elem(b, 0, o)? as u32;
                for l in 0..num_layers {
                    unsafe {
                        *bases_ptr.add(l * stride + pos) = base_l0 + l as u32 * elems_per_layer;
                    }
                }
            }
        } else {
            let mut token_bases_l0: Vec<u32> = Vec::with_capacity(seq);
            for t in 0..seq {
                let b = page_table.block_for_token(t)
                    .map_err(|e| CoreError::Backend(format!("prefill bases: {e}")))?;
                let o = page_table.offset_in_block(t)
                    .map_err(|e| CoreError::Backend(format!("prefill bases: {e}")))?;
                token_bases_l0.push(layout.token_base_elem(b, 0, o)? as u32);
            }
            for l in 0..num_layers {
                let layer_add = l as u32 * elems_per_layer;
                let row_ptr = unsafe { bases_ptr.add(l * stride) };
                for t in 0..seq {
                    unsafe { *row_ptr.add(t) = token_bases_l0[t] + layer_add; }
                }
            }
        }

        self.bases_last_session = session_id;
        self.bases_last_seq = seq;
        let bases_ref = self.bases_buf.as_ref().unwrap();

        // ---- 2. Upload ALL token embeddings to a GPU buffer at once ----
        let all_x_bytes = (x_all.len() * 4) as u64;
        let all_x_buf = self.device.new_buffer_with_data(
            x_all.as_ptr() as *const std::ffi::c_void,
            all_x_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        // ---- 3. Pre-compute layer weight names ----
        struct LayerWeightNames {
            in_norm: [String; 2],
            q_proj: [String; 2],
            k_proj: [String; 2],
            v_proj: [String; 2],
            o_proj: [String; 2],
            post_norm: [String; 2],
            gate_proj: [String; 2],
            up_proj: [String; 2],
            down_proj: [String; 2],
        }
        let layer_names: Vec<LayerWeightNames> = (0..num_layers).map(|l| {
            let mk = |name: &str| -> [String; 2] {
                [format!("{prefix}{name}"), name.to_string()]
            };
            LayerWeightNames {
                in_norm: mk(&format!("model.layers.{l}.input_layernorm.weight")),
                q_proj: mk(&format!("model.layers.{l}.self_attn.q_proj.weight")),
                k_proj: mk(&format!("model.layers.{l}.self_attn.k_proj.weight")),
                v_proj: mk(&format!("model.layers.{l}.self_attn.v_proj.weight")),
                o_proj: mk(&format!("model.layers.{l}.self_attn.o_proj.weight")),
                post_norm: mk(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                gate_proj: mk(&format!("model.layers.{l}.mlp.gate_proj.weight")),
                up_proj: mk(&format!("model.layers.{l}.mlp.up_proj.weight")),
                down_proj: mk(&format!("model.layers.{l}.mlp.down_proj.weight")),
            }
        }).collect();

        // ---- 4. Allocate per-token Q buffer for batched attention ----
        let all_tokens_bytes = (num_tokens * hidden * 4) as u64;
        let q_all_buf = self.device.new_buffer(all_tokens_bytes, MTLResourceOptions::StorageModeShared);
        let attn_out_all_buf = self.device.new_buffer(all_tokens_bytes, MTLResourceOptions::StorageModeShared);

        // ---- 5. Single compute encoder per layer (per-token MV + batched attention) ----
        autoreleasepool(|| -> Result<Option<Vec<f32>>, CoreError> {
        let cb = self.queue.new_command_buffer();

        for layer in 0..num_layers {
            let ln = &layer_names[layer];
            let enc = cb.new_compute_command_encoder();

            // Pre-attention: per-token RMS norm, QKV, RoPE, write KV, save Q to q_all
            for token_idx in 0..num_tokens {
                let pos = start_pos + token_idx;
                let block_id = page_table.block_for_token(pos)
                    .map_err(|e| CoreError::Backend(format!("prefill block: {e}")))?;
                let token_off = page_table.offset_in_block(pos)
                    .map_err(|e| CoreError::Backend(format!("prefill offset: {e}")))?;
                let tok_off = token_idx * hidden;

                self.ops.encode_copy_f32(&enc, &all_x_buf, &self.x_buf, tok_off, 0, hidden);
                let w_in = self.get_weight(&ln.in_norm[0], Some(&ln.in_norm[1]));
                self.ops.encode_rms_norm_f16w(&enc, &self.x_buf, w_in, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

                // Per-token QKV
                let w_q = self.get_weight(&ln.q_proj[0], Some(&ln.q_proj[1]));
                let w_k = self.get_weight(&ln.k_proj[0], Some(&ln.k_proj[1]));
                let w_v = self.get_weight(&ln.v_proj[0], Some(&ln.v_proj[1]));
                self.ops.encode_mv_f16(&enc, w_q, &self.x_norm_buf, &self.q_buf, hidden, hidden);
                let kv_dim = n_kv_heads * head_dim;
                self.ops.encode_mv_f16(&enc, w_k, &self.x_norm_buf, &self.k_buf, kv_dim, hidden);
                self.ops.encode_mv_f16(&enc, w_v, &self.x_norm_buf, &self.v_buf, kv_dim, hidden);

                // RoPE
                if self.rope_interleaved {
                    self.ops.encode_rope_adj_f32(&enc, &self.q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                    self.ops.encode_rope_adj_f32(&enc, &self.k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
                } else {
                    self.ops.encode_rope_half_f32(&enc, &self.q_buf, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                    self.ops.encode_rope_half_f32(&enc, &self.k_buf, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
                }

                // Write KV
                let kv_store = kv_cache.storage().as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                    .expect("prefill requires MetalKvStorage");
                let target_base = kv_cache.layout().token_base_elem(block_id, layer, token_off)
                    .map_err(|e| CoreError::Backend(format!("prefill base_elem: {e}")))?;
                kv_store.encode_write_token_f32(&enc, target_base, &self.k_buf, &self.v_buf, kv_dim);

                // Save Q for batched attention
                self.ops.encode_copy_f32(&enc, &self.q_buf, &q_all_buf, 0, tok_off, hidden);
            }

            // Batched attention
            let bases_offset = (layer * stride * 4) as u64;
            let kv_store = kv_cache.storage().as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                .expect("prefill requires MetalKvStorage");
            kv_store.encode_attention_batched(
                &enc, bases_ref, bases_offset, &q_all_buf, &attn_out_all_buf,
                start_pos as u32, num_tokens as u32,
                n_heads as u32, n_kv_heads as u32, head_dim as u32,
                None, None,
            );

            // Post-attention: per-token O projection, residual, MLP
            for token_idx in 0..num_tokens {
                let tok_off = token_idx * hidden;

                // O projection + residual
                self.ops.encode_copy_f32(&enc, &all_x_buf, &self.x_buf, tok_off, 0, hidden);
                self.ops.encode_copy_f32(&enc, &attn_out_all_buf, &self.attn_out_buf, tok_off, 0, hidden);
                let w_o = self.get_weight(&ln.o_proj[0], Some(&ln.o_proj[1]));
                self.ops.encode_mv_f16(&enc, w_o, &self.attn_out_buf, &self.mlp_in_buf, hidden, hidden);
                self.ops.encode_add_f32_inplace(&enc, &self.x_buf, &self.mlp_in_buf, hidden);

                // Post-norm
                let w_post = self.get_weight(&ln.post_norm[0], Some(&ln.post_norm[1]));
                self.ops.encode_rms_norm_f16w(&enc, &self.x_buf, w_post, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

                // Gate + Up
                let w_gate = self.get_weight(&ln.gate_proj[0], Some(&ln.gate_proj[1]));
                let w_up   = self.get_weight(&ln.up_proj[0], Some(&ln.up_proj[1]));
                self.ops.encode_mv_f16(&enc, w_gate, &self.x_norm_buf, &self.gate_buf, cfg.intermediate_size, hidden);
                self.ops.encode_mv_f16(&enc, w_up,   &self.x_norm_buf, &self.up_buf,   cfg.intermediate_size, hidden);
                self.ops.encode_silu_mul_f32_inplace(&enc, &self.gate_buf, &self.up_buf, cfg.intermediate_size);

                // Down + residual
                let w_down = self.get_weight(&ln.down_proj[0], Some(&ln.down_proj[1]));
                self.ops.encode_mv_f16(&enc, w_down, &self.gate_buf, &self.down_buf, hidden, cfg.intermediate_size);
                self.ops.encode_add_f32_inplace(&enc, &self.x_buf, &self.down_buf, hidden);

                self.ops.encode_copy_f32(&enc, &self.x_buf, &all_x_buf, 0, tok_off, hidden);
            }

            enc.end_encoding();
        }

        // ---- Final norm + LM head for the last token (only if return_logits) ----
        // After all layers, x_buf contains the last token's final layer output.
        let maybe_logits = if return_logits {
            let enc_final = cb.new_compute_command_encoder();
            let w_norm = self.get_weight(
                &format!("{prefix}model.norm.weight"),
                Some("model.norm.weight"),
            );
            self.ops.encode_rms_norm_f16w(
                &enc_final, &self.x_buf, w_norm, &self.x_norm_buf,
                hidden, cfg.rms_norm_eps, false,
            );

            let lm_head_name = format!("{prefix}lm_head.weight");
            let embed_name = format!("{prefix}model.embed_tokens.weight");
            let lm_weight_name = if self.weights.contains_key(&lm_head_name) {
                lm_head_name
            } else if self.weights.contains_key("lm_head.weight") {
                "lm_head.weight".to_string()
            } else if self.weights.contains_key(&embed_name) {
                embed_name
            } else {
                "model.embed_tokens.weight".to_string()
            };
            let w_lm = self.weights.get(&lm_weight_name).expect("LM head not found");
            if let Some(s_lm) = self.try_get_weight(&format!("{}.qscale", lm_weight_name)) {
                self.ops.encode_mv_i8(&enc_final, w_lm, s_lm, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            } else {
                self.ops.encode_mv_f16(&enc_final, w_lm, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            }
            enc_final.end_encoding();

            cb.commit();
            cb.wait_until_completed();

            let mut logits = vec![0.0f32; cfg.vocab_size];
            unsafe {
                let ptr = self.logits_buf.contents() as *const f32;
                std::ptr::copy_nonoverlapping(ptr, logits.as_mut_ptr(), cfg.vocab_size);
            }
            Some(logits)
        } else {
            cb.commit();
            cb.wait_until_completed();
            None
        };

        Ok(maybe_logits)
        }) // autoreleasepool
    }

    pub fn preload_weight(&mut self, name: String, bytes: &[u8]) {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.weights.insert(name, buf);
    }

    pub fn preload_weight_f16(&mut self, name: String, bytes: &[u8]) {
        self.preload_weight(name, bytes);
    }

    fn get_weight(&self, name: &str, alt: Option<&str>) -> &Buffer {
        if let Some(w) = self.weights.get(name) { return w; }
        if let Some(a) = alt {
            if let Some(w) = self.weights.get(a) { return w; }
        }
        // Fallback: try removing 'model.' prefix if it was added,
        // and also try the 'model.text_model.' prefix which some converted
        // checkpoints use (e.g., smolvlm models). This makes the loader robust
        // to both naming conventions.
        if name.starts_with("model.") {
            // Strip the leading "model."
            let stripped = &name[6..];
            if let Some(w) = self.weights.get(stripped) { return w; }
            // Try with "model.text_model." prefix added back
            let txt_name = format!("model.text_model.{}", stripped);
            if let Some(w) = self.weights.get(&txt_name) { return w; }
        }
        panic!("Weight not found: {} (alt: {:?})", name, alt);
    }

    fn try_get_weight(&self, name: &str) -> Option<&Buffer> {
        self.weights.get(name).or_else(|| {
            if name.starts_with("model.") {
                // Strip the leading "model."
                let stripped = &name[6..];
                // First try the raw stripped name
                if let Some(w) = self.weights.get(stripped) { return Some(w); }
                // Then try the "model.text_model." prefixed version
                let txt_name = format!("model.text_model.{}", stripped);
                return self.weights.get(&txt_name);
            }
            None
        })
    }

    pub fn step_fused(
        &mut self,
        x_in: &[f32],
        cfg: &crate::ModelConfig,
        prefix: &str,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        pos: usize,
        token_off: usize,
        block_id: u32,
        return_logits: bool,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        autoreleasepool(|| {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden / n_heads;
        let _kv_dim = n_kv_heads * head_dim;

        // 1. Setup execution state, send X embeddings to Metal
        unsafe {
            let ptr = self.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x_in.as_ptr(), ptr, hidden);
        }

        // 2. Build KV-bases index for the attention kernels.
        //
        // Layout: [num_layers][bases_stride] u32, where bases_stride is a fixed
        // power-of-two capacity.  Layer l's entries live at byte offset
        // l * bases_stride * 4.  This stable layout allows incremental updates:
        // on consecutive decode steps for the same session we only need to write
        // num_layers new u32 values (one per layer) rather than rebuilding the
        // entire seq × num_layers array.
        //
        // Decomposed formula (avoids O(seq × L) page-table lookups):
        //   base(t, layer) = base(t, 0) + layer * elems_per_block_per_layer
        let seq        = page_table.token_count(); // pos + 1
        let num_layers = cfg.num_hidden_layers;
        let session_id = page_table.session_id();

        // Grow buffer if the current stride is too small for `seq`.
        let need_resize = self.bases_stride < seq;
        if need_resize || self.bases_buf.is_none() {
            let new_stride = seq.next_power_of_two().max(64);
            self.bases_stride = new_stride;
            self.bases_buf = Some(self.device.new_buffer(
                (new_stride * num_layers * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.bases_last_seq = 0; // force full rebuild after resize
        }

        // SAFETY: StorageModeShared buffers are CPU-accessible; no GPU work is
        // in flight (we haven't opened the command buffer yet).
        let bases_ptr = self.bases_buf.as_ref().unwrap().contents() as *mut u32;
        let layout    = kv_cache.layout(); // KvCacheLayout is Copy
        let elems_per_layer = layout.elems_per_block_per_layer() as u32;
        let stride = self.bases_stride;

        let incremental = !need_resize
            && self.bases_last_session == session_id
            && self.bases_last_seq + 1 == seq;

        if incremental {
            // Fast path: only write the one new token's entry per layer. O(num_layers).
            let b = page_table.block_for_token(pos)
                .map_err(|e| CoreError::Backend(format!("graph bases: {e}")))?;
            let o = page_table.offset_in_block(pos)
                .map_err(|e| CoreError::Backend(format!("graph bases: {e}")))?;
            let base_l0 = layout.token_base_elem(b, 0, o)? as u32;
            for l in 0..num_layers {
                unsafe {
                    *bases_ptr.add(l * stride + pos) = base_l0 + l as u32 * elems_per_layer;
                }
            }
        } else {
            // Full rebuild: O(seq) page-table lookups + O(seq × L) cheap additions.
            let mut token_bases_l0: Vec<u32> = Vec::with_capacity(seq);
            for t in 0..seq {
                let b = page_table.block_for_token(t)
                    .map_err(|e| CoreError::Backend(format!("graph bases: {e}")))?;
                let o = page_table.offset_in_block(t)
                    .map_err(|e| CoreError::Backend(format!("graph bases: {e}")))?;
                token_bases_l0.push(layout.token_base_elem(b, 0, o)? as u32);
            }
            for l in 0..num_layers {
                let layer_add = l as u32 * elems_per_layer;
                let row_ptr = unsafe { bases_ptr.add(l * stride) };
                for t in 0..seq {
                    unsafe { *row_ptr.add(t) = token_bases_l0[t] + layer_add; }
                }
            }
        }

        self.bases_last_session = session_id;
        self.bases_last_seq = seq;

        let bases_ref = self.bases_buf.as_ref().unwrap();

        // 3. Initiate Command Buffer!
        let cb = self.queue.new_command_buffer();

        // 4. Single compute encoder for all layers — dispatches execute serially
        // within an encoder, eliminating ~24× CPU overhead from create/destroy.
        let enc = cb.new_compute_command_encoder();
        for layer in 0..num_layers {
            let w_in_norm = self.get_weight(&format!("{prefix}model.layers.{layer}.input_layernorm.weight"), Some(&format!("model.layers.{layer}.input_layernorm.weight")));
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_in_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

            let w_q = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.q_proj.weight"));
            let w_k = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.k_proj.weight"));
            let w_v = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.v_proj.weight"));

            if let (Some(wq), Some(wk), Some(wv)) = (w_q, w_k, w_v) {
                let s_q = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.q_proj.weight.qscale"))
                    .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.q_proj.qscale")));
                let s_k = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.k_proj.weight.qscale"))
                    .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.k_proj.qscale")));
                let s_v = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.v_proj.weight.qscale"))
                    .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.v_proj.qscale")));

                if let (Some(sq), Some(sk), Some(sv)) = (s_q, s_k, s_v) {
                    self.ops.encode_qkv_i8(enc, wq, sq, wk, sk, wv, sv, &self.x_norm_buf, &self.q_buf, &self.k_buf, &self.v_buf, n_heads * head_dim, n_kv_heads * head_dim, hidden);
                } else {
                    self.ops.encode_qkv_f16(enc, wq, wk, wv, &self.x_norm_buf, &self.q_buf, &self.k_buf, &self.v_buf, n_heads * head_dim, n_kv_heads * head_dim, hidden);
                }
            } else {
                return Err(CoreError::Backend(format!("missing Q/K/V weights for layer {layer}")));
            }

            if self.rope_interleaved {
                self.ops.encode_rope_adj_f32(enc, &self.q_buf, n_heads, head_dim, pos, cfg.rope_theta);
                self.ops.encode_rope_adj_f32(enc, &self.k_buf, n_kv_heads, head_dim, pos, cfg.rope_theta);
            } else {
                // rotate-half style (SmolLM2, Mistral, etc.)
                self.ops.encode_rope_half_f32(enc, &self.q_buf, n_heads, head_dim, head_dim, pos, cfg.rope_theta);
                self.ops.encode_rope_half_f32(enc, &self.k_buf, n_kv_heads, head_dim, head_dim, pos, cfg.rope_theta);
            }

            let kv_store = kv_cache.storage().as_any().downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                .expect("fused graph requires MetalKvStorage");

            let target_base = kv_cache.layout().token_base_elem(block_id, layer, token_off)?;
            kv_store.encode_write_token_f32(enc, target_base, &self.k_buf, &self.v_buf, n_kv_heads * head_dim);

            let bases_offset = (layer * stride * 4) as u64;
            kv_store.encode_attention(
                enc,
                bases_ref,
                bases_offset,
                &self.q_buf,
                &self.attn_out_buf,
                seq as u32,    // fixed: full context length, was wrongly hardcoded to 1
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                None,
                None,
            );

            let w_o = self.get_weight(&format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight"), Some(&format!("model.layers.{layer}.self_attn.o_proj.weight")));
            if let Some(s_o) = self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.o_proj.weight.qscale"))
                .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.self_attn.o_proj.qscale"))) {
                self.ops.encode_mv_i8(enc, w_o, s_o, &self.attn_out_buf, &self.mlp_in_buf, hidden, hidden);
            } else {
                self.ops.encode_mv_f16(enc, w_o, &self.attn_out_buf, &self.mlp_in_buf, hidden, hidden);
            }

            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.mlp_in_buf, hidden);

            let w_post_norm = self.get_weight(&format!("{prefix}model.layers.{layer}.post_attention_layernorm.weight"), Some(&format!("model.layers.{layer}.post_attention_layernorm.weight")));
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_post_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

            let w_gate = self.get_weight(&format!("{prefix}model.layers.{layer}.mlp.gate_proj.weight"), Some(&format!("model.layers.{layer}.mlp.gate_proj.weight")));
            let w_up = self.get_weight(&format!("{prefix}model.layers.{layer}.mlp.up_proj.weight"), Some(&format!("model.layers.{layer}.mlp.up_proj.weight")));

            let s_gate = self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.gate_proj.weight.qscale"))
                .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.gate_proj.qscale")));
            let s_up = self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.up_proj.weight.qscale"))
                .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.up_proj.qscale")));

            if let (Some(sg), Some(su)) = (s_gate, s_up) {
                self.ops.encode_mv_i8(enc, w_gate, sg, &self.x_norm_buf, &self.gate_buf, cfg.intermediate_size, hidden);
                self.ops.encode_mv_i8(enc, w_up, su, &self.x_norm_buf, &self.up_buf, cfg.intermediate_size, hidden);
            } else {
                self.ops.encode_mv_f16(enc, w_gate, &self.x_norm_buf, &self.gate_buf, cfg.intermediate_size, hidden);
                self.ops.encode_mv_f16(enc, w_up, &self.x_norm_buf, &self.up_buf, cfg.intermediate_size, hidden);
            }

            self.ops.encode_silu_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, cfg.intermediate_size);

            let w_down = self.get_weight(&format!("{prefix}model.layers.{layer}.mlp.down_proj.weight"), Some(&format!("model.layers.{layer}.mlp.down_proj.weight")));
            let s_down = self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.down_proj.weight.qscale"))
                .or_else(|| self.try_get_weight(&format!("{prefix}model.layers.{layer}.mlp.down_proj.qscale")));

            if let Some(sd) = s_down {
                self.ops.encode_mv_i8(enc, w_down, sd, &self.gate_buf, &self.down_buf, hidden, cfg.intermediate_size);
            } else {
                self.ops.encode_mv_f16(enc, w_down, &self.gate_buf, &self.down_buf, hidden, cfg.intermediate_size);
            }

            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.down_buf, hidden);
        }
        enc.end_encoding();

        // 5. Final norm + logits in the same command buffer.
        let enc = cb.new_compute_command_encoder();
        let w_norm = self.get_weight(&format!("{prefix}model.norm.weight"), Some("model.norm.weight"));
        self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

        let lm_head_name = format!("{prefix}lm_head.weight");
        let embed_name = format!("{prefix}model.embed_tokens.weight");

        if return_logits {
            let lm_weight_name = if self.weights.contains_key(&lm_head_name) {
                lm_head_name
            } else if self.weights.contains_key("lm_head.weight") {
                "lm_head.weight".to_string()
            } else if self.weights.contains_key(&embed_name) {
                embed_name
            } else {
                "model.embed_tokens.weight".to_string()
            };

            let w_lm = self.weights.get(&lm_weight_name).expect("LM head not found");
            if let Some(s_lm) = self.try_get_weight(&format!("{}.qscale", lm_weight_name)) {
                self.ops.encode_mv_i8(enc, w_lm, s_lm, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            } else {
                self.ops.encode_mv_f16(enc, w_lm, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
            }
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        if !return_logits {
            return Ok(None);
        }

        let mut logits = vec![0.0f32; cfg.vocab_size];

        unsafe {
            let ptr = self.logits_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, logits.as_mut_ptr(), cfg.vocab_size);
        }

        let mut nan_count = 0;
        let mut inf_count = 0;
        for &v in &logits {
            if v.is_nan() { nan_count += 1; }
            else if v.is_infinite() { inf_count += 1; }
        }

        if nan_count > 0 || inf_count > 0 {
            return Err(CoreError::Backend(format!("LlamaGraphState: divergence detected at pos {pos} (NaNs={nan_count}, Infs={inf_count})")));
        }

        Ok(Some(logits))
        })
    }
}
