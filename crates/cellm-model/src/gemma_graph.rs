// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Gemma3 fused Metal graph — single command-buffer decode.
//!
//! Handles the 4-norm branch-residual structure, per-head Q/K norms,
//! GELU activation, rotate-half RoPE, and sliding-window attention.

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

/// Per-layer attention geometry stored in the graph state.
#[derive(Clone, Debug)]
pub struct GemmaGraphLayerSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
}

/// Fused Metal inference graph for Gemma3 text models.
///
/// Encodes one full forward pass (all layers) into a single `CommandBuffer`
/// with a single `wait_until_completed` sync point, matching the Llama graph approach.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct GemmaGraphState {
    pub device: Device,
    pub queue: CommandQueue,
    pub ops: MetalOps,

    /// Model weights pre-uploaded to Metal shared memory.
    pub weights: HashMap<String, Buffer>,
    /// Dtype string per tensor name ("f16" | "i8").
    pub tensor_dtypes: HashMap<String, String>,

    // Persistent activation buffers.
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
    /// V-norm ones buffer for Gemma4 RMSNorm without weight (f16 ones).
    pub v_norm_ones_buf: Buffer,

    /// KV-bases index buffer, fixed-stride layout [num_layers][bases_stride] u32.
    pub bases_buf: Option<Buffer>,
    pub bases_stride: usize,       // allocated tokens-per-layer (power of two, >= seq)
    pub bases_last_session: u64,   // session that last wrote to this buffer
    pub bases_last_seq: usize,     // token_count when buffer was last updated

    // Static model properties.
    pub layer_specs: Vec<GemmaGraphLayerSpec>,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub is_gemma3: bool,
    pub is_gemma4: bool,
    pub gemma4_shared_kv_layers: usize,
    pub gemma4_sliding_mask: Vec<bool>,

    // PLE (Per-Layer Embedding input) buffer
    pub ple_buf: Option<Buffer>,
    /// Temporary buffer for current layer's PLE input slice (GPU).
    pub ple_tmp_buf: Option<Buffer>,
    /// Scratch buffer for layer output scale f32 data (GPU).
    pub layer_scale_buf: Option<Buffer>,
    pub ple_per_layer_dim: usize,
    pub rope_theta_sliding: f32,
    pub rmsnorm_weight_is_offset: bool,
    pub tensor_prefix: String,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl GemmaGraphState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        max_q_dim: usize,
        max_kv_dim: usize,
        max_ffn_dim: usize,
        layer_specs: Vec<GemmaGraphLayerSpec>,
        sliding_window: usize,
        sliding_window_pattern: usize,
        is_gemma3: bool,
        is_gemma4: bool,
        gemma4_shared_kv_layers: usize,
        gemma4_sliding_mask: Vec<bool>,
        rope_theta_sliding: f32,
        rmsnorm_weight_is_offset: bool,
        tensor_prefix: String,
        ops: MetalOps,
        ple_per_layer_dim: usize,
    ) -> Result<Self, CoreError> {
        let device = ops.device.clone();
        let queue = ops.queue.clone();

        let mk = |n: usize| {
            device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
        };

        // Pre-compute all buffers before the struct literal so the `mk` borrow of
        // `device` ends before `device` is moved into `Self`.
        let x_buf        = mk(hidden_size);
        let x_norm_buf   = mk(hidden_size);
        let q_buf        = mk(max_q_dim);
        let k_buf        = mk(max_kv_dim);
        let v_buf        = mk(max_kv_dim);
        let attn_out_buf = mk(max_q_dim);
        let mlp_in_buf   = mk(hidden_size);
        let gate_buf     = mk(max_ffn_dim);
        let up_buf       = mk(max_ffn_dim);
        let down_buf     = mk(hidden_size);
        let logits_buf   = mk(vocab_size);
        let _ = mk; // release borrow of `device` before it is moved

        // Create V-norm ones buffer for Gemma4 (per-head RMSNorm without weight).
        let max_kv_head_dim = layer_specs.iter()
            .map(|s| s.kv_head_dim)
            .max()
            .unwrap_or(64);
        let v_norm_ones_buf = {
            let ones: Vec<u16> = vec![0x3C00u16; max_kv_head_dim]; // 1.0 in f16
            device.new_buffer_with_data(
                ones.as_ptr() as *const std::ffi::c_void,
                (max_kv_head_dim * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Allocate PLE temporary buffer (per-layer slice) if needed.
        let ple_tmp_buf = if ple_per_layer_dim > 0 {
            let buf = device.new_buffer(
                (ple_per_layer_dim * 4) as u64,  // f32 elements
                MTLResourceOptions::StorageModeShared,
            );
            Some(buf)
        } else {
            None
        };

        // Allocate scratch buffer for layer output scale (f32, hidden_size elements).
        let layer_scale_buf = if is_gemma4 {
            let buf = device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            Some(buf)
        } else {
            None
        };

        // Allocate ple_buf (the big one for all per-layer data).
        let full_ple_buf = if ple_per_layer_dim > 0 && !layer_specs.is_empty() {
            let total_ple = layer_specs.len() * ple_per_layer_dim;
            let buf = device.new_buffer(
                (total_ple * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            Some(buf)
        } else {
            None
        };

        Ok(Self {
            device,
            queue,
            ops,
            weights: HashMap::new(),
            tensor_dtypes: HashMap::new(),
            x_buf,
            x_norm_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            mlp_in_buf,
            gate_buf,
            up_buf,
            down_buf,
            logits_buf,
            v_norm_ones_buf,
            bases_buf: None,
            bases_stride: 0,
            bases_last_session: 0,
            bases_last_seq: 0,
            layer_specs,
            sliding_window,
            sliding_window_pattern,
            is_gemma3,
            is_gemma4,
            gemma4_shared_kv_layers,
            gemma4_sliding_mask,
            ple_buf: full_ple_buf,
            ple_tmp_buf,
            layer_scale_buf,
            ple_per_layer_dim,
            rope_theta_sliding,
            rmsnorm_weight_is_offset,
            tensor_prefix,
        })
    }

    pub fn preload_weight(&mut self, name: String, bytes: &[u8], dtype: String) {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.tensor_dtypes.insert(name.clone(), dtype);
        self.weights.insert(name, buf);
    }

    fn get_w(&self, name: &str) -> &Buffer {
        self.weights.get(name)
            .or_else(|| if name.starts_with("model.") { self.weights.get(&name[6..]) } else { None })
            .or_else(|| self.weights.get(&format!("model.{name}")))
            // For Gemma4 models with model.language_model.layers prefix
            .or_else(|| {
                if name.starts_with("model.") {
                    let lm = format!("model.language_model.{}", &name[6..]);
                    self.weights.get(&lm)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("GemmaGraph: weight not found: {name}"))
    }

    fn try_get_w(&self, name: &str) -> Option<&Buffer> {
        self.weights.get(name)
            .or_else(|| if name.starts_with("model.") { self.weights.get(&name[6..]) } else { None })
            .or_else(|| self.weights.get(&format!("model.{name}")))
            // For Gemma4 models with model.language_model.layers prefix
            .or_else(|| {
                if name.starts_with("model.") {
                    let lm = format!("model.language_model.{}", &name[6..]);
                    self.weights.get(&lm)
                } else {
                    None
                }
            })
    }

    /// Resolve weight dtype with prefix-aware lookup (same strategy as get_w).
    fn try_get_dtype(&self, name: &str) -> Option<&str> {
        self.tensor_dtypes.get(name)
            .or_else(|| if name.starts_with("model.") { self.tensor_dtypes.get(&name[6..]) } else { None })
            .or_else(|| {
                if name.starts_with("model.") {
                    let lm = format!("model.language_model.{}", &name[6..]);
                    self.tensor_dtypes.get(&lm)
                } else {
                    None
                }
            })
            .map(|s| s.as_str())
    }

    /// For Gemma4 shared-KV layers: returns the source layer index whose KV
    /// cache to use, or `None` if this layer manages its own KV.
    fn gemma4_kv_source_layer(&self, layer: usize) -> Option<usize> {
        if !self.is_gemma4 || self.gemma4_shared_kv_layers == 0 {
            return None;
        }
        let max_layers = self.layer_specs.len();
        let first_shared = max_layers.saturating_sub(self.gemma4_shared_kv_layers);
        if layer < first_shared {
            return None;
        }
        let is_sliding = self.gemma4_sliding_mask.get(layer).copied().unwrap_or(false);
        (0..first_shared)
            .rev()
            .find(|&i| self.gemma4_sliding_mask.get(i).copied().unwrap_or(false) == is_sliding)
    }

    /// Whether layer `l` uses full (global) attention.
    /// Gemma3 pattern: every `sliding_window_pattern`-th layer (1-indexed) is full-attention.
    /// Gemma4: uses a boolean mask where `true` = sliding-window, `false` = full attention.
    fn is_full_attention_layer(&self, layer: usize) -> bool {
        if self.is_gemma4 {
            // Gemma4: mask=true means sliding-window, false means full attention
            !self.gemma4_sliding_mask.get(layer).copied().unwrap_or(true)
        } else if self.is_gemma3 {
            // Gemma3 pattern: 5 sliding layers, then 1 full-attention layer
            self.sliding_window_pattern != 0 && (layer + 1) % self.sliding_window_pattern == 0
        } else {
            true
        }
    }

    /// Single-token fused decode.
    ///
    /// `token_off` and `block_id` are for the current token position (already appended
    /// to the page table by the caller).  `pos` is the 0-based sequence position.
    ///
    /// Returns `Some(raw_logits)` when `return_logits=true`, `None` otherwise.
    pub fn step_fused(
        &mut self,
        x_in: &[f32],
        cfg: &crate::ModelConfig,
        kv_cache: &mut KVCache,
        page_table: &PageTable,
        pos: usize,
        token_off: usize,
        block_id: u32,
        return_logits: bool,
        per_layer_input: Option<&[f32]>,
    ) -> Result<Option<Vec<f32>>, CoreError> {
        let hidden = cfg.hidden_size;
        let num_layers = cfg.num_hidden_layers;
        let prefix = self.tensor_prefix.clone();
        let add_one = self.rmsnorm_weight_is_offset;

        // Upload PLE data to GPU buffer for per-layer GPU PLE injection.
        let ple_data: Option<(Vec<f32>, usize)> = match per_layer_input {
            Some(data) if self.is_gemma4 && self.ple_per_layer_dim > 0 && !data.is_empty() => {
                let ppl = self.ple_per_layer_dim;
                // Upload to GPU buffer at the start (all layers at once).
                if let Some(ref ple_buf) = self.ple_buf {
                    let copy_len = data.len().min(ple_buf.length() as usize / 4);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            ple_buf.contents() as *mut f32,
                            copy_len,
                        );
                    }
                }
                Some((data.to_vec(), ppl))
            }
            _ => None,
        };

        // 1. Upload embedding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_in.as_ptr(),
                self.x_buf.contents() as *mut f32,
                hidden,
            );
        }

        // 2. Build KV-bases index for attention kernels.
        //
        // Fixed-stride layout: [num_layers][bases_stride] u32.
        // Layer l's slot starts at byte offset l * bases_stride * 4.
        //
        // For full-attention layers:  bases_off = l * stride * 4,  count = seq
        // For sliding-window layers:  bases_off = (l * stride + start_tpos) * 4,
        //                             count = min(seq, sliding_window)
        //
        // Incremental: on consecutive steps for the same session we only write
        // num_layers new entries (one per layer for token `pos`). O(num_layers).
        // Full rebuild uses the decomposed formula:
        //   base(t, l) = base(t, 0) + l * elems_per_block_per_layer
        // reducing page-table lookups from O(seq × L) to O(seq).
        let seq        = page_table.token_count();
        let session_id = page_table.session_id();

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

        let bases_ptr       = self.bases_buf.as_ref().unwrap().contents() as *mut u32;
        let layout          = kv_cache.layout();
        let elems_per_layer = layout.elems_per_block_per_layer() as u32;
        let stride          = self.bases_stride;

        let incremental = !need_resize
            && self.bases_last_session == session_id
            && self.bases_last_seq + 1 == seq;

        if incremental {
            // Write only the new token's entry per layer. O(num_layers).
            let b = page_table.block_for_token(pos)
                .map_err(|e| CoreError::Backend(format!("gemma graph bases: {e}")))?;
            let o = page_table.offset_in_block(pos)
                .map_err(|e| CoreError::Backend(format!("gemma graph bases: {e}")))?;
            let base_l0 = layout.token_base_elem(b, 0, o)? as u32;
            for l in 0..num_layers {
                unsafe {
                    *bases_ptr.add(l * stride + pos) = base_l0 + l as u32 * elems_per_layer;
                }
            }
        } else {
            // Full rebuild: O(seq) lookups, O(seq × L) additions.
            let mut token_bases_l0: Vec<u32> = Vec::with_capacity(seq);
            for t in 0..seq {
                let b = page_table.block_for_token(t)
                    .map_err(|e| CoreError::Backend(format!("gemma graph bases l0: {e}")))?;
                let o = page_table.offset_in_block(t)
                    .map_err(|e| CoreError::Backend(format!("gemma graph bases l0: {e}")))?;
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

        // 3. Initiate Command Buffer.
        let cb = self.queue.new_command_buffer();

        // 4. One encoder per layer — ensures memory ordering between layers.
        for layer in 0..num_layers {
            // Pre-load layer output scale into layer_scale_buf for this layer
            // (needed by GPU PLE section 4q).  Scale is f16; we convert to f32
            // on CPU since the buffer is StorageModeShared.
            if self.is_gemma4 && self.layer_scale_buf.is_some() {
                let has_ple = ple_data.as_ref()
                    .map(|&(ref ple_in, ppl)| ppl > 0 && layer * ppl + ppl <= ple_in.len())
                    .unwrap_or(false);
                if has_ple {
                    let pref = format!("{prefix}model.layers.{layer}");
                    let scale_candidates = [
                        format!("{pref}.layer_output_scale.weight"),
                        format!("{pref}.layer_scalar"),
                    ];
                    for scale_name in &scale_candidates {
                        if let Some(scale_buf) = self.try_get_w(&scale_name) {
                            let scale_data = unsafe {
                                std::slice::from_raw_parts(
                                    scale_buf.contents() as *const u16,
                                    scale_buf.length() as usize / 2,
                                )
                            };
                            if let Some(ref ls_buf) = self.layer_scale_buf {
                                let ls_ptr = ls_buf.contents() as *mut f32;
                                if scale_data.len() == 1 {
                                    let s = half::f16::from_bits(scale_data[0]).to_f32();
                                    for i in 0..hidden {
                                        unsafe { *ls_ptr.add(i) = s; }
                                    }
                                } else if scale_data.len() == hidden {
                                    for i in 0..hidden {
                                        let s = half::f16::from_bits(scale_data[i]).to_f32();
                                        unsafe { *ls_ptr.add(i) = s; }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }

            let enc = cb.new_compute_command_encoder();
            let spec = self.layer_specs[layer].clone();
            let n_heads     = spec.n_heads;
            let n_kv_heads  = spec.n_kv_heads;
            let head_dim    = spec.head_dim;
            let kv_head_dim = spec.kv_head_dim;
            let q_dim       = spec.q_dim;
            let kv_dim      = spec.kv_dim;
            let ffn_dim     = spec.ffn_dim;
            let is_full     = self.is_full_attention_layer(layer);
            let start_tpos  = if is_full { 0 } else { seq.saturating_sub(self.sliding_window) };
            let attn_count  = (seq - start_tpos) as u32;
            let rope_theta  = if is_full { cfg.rope_theta } else { self.rope_theta_sliding };

            let pref = format!("{prefix}model.layers.{layer}");
            // For Gemma4 shared-KV layers: use source layer's cache slot for attention
            let is_kv_shared = self.gemma4_kv_source_layer(layer).is_some();
            let kv_src_layer = self.gemma4_kv_source_layer(layer).unwrap_or(layer);
            // byte offset into bases_buf for this layer's slot (sliding-window aware)
            let bases_off   = ((kv_src_layer * stride + start_tpos) * 4) as u64;

            // 4a. Input RMSNorm: x → x_norm
            {
                let w = self.get_w(&format!("{pref}.input_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4b. QKV projections (skip K/V for shared-KV layers).
            {
                let wq = self.get_w(&format!("{pref}.self_attn.q_proj.weight"));
                let q_dtype = self.try_get_dtype(&format!("{pref}.self_attn.q_proj.weight"))
                    .unwrap_or("f16");
                if q_dtype == "i4" {
                    let sq = self.get_w(&format!("{pref}.self_attn.q_proj.weight.qscale"));
                    self.ops.encode_mv_i4(enc, wq, sq, &self.x_norm_buf, &self.q_buf, q_dim, hidden, hidden);
                } else {
                    let sq = self.try_get_w(&format!("{pref}.self_attn.q_proj.weight.qscale"));
                    if let Some(sq) = sq {
                        self.ops.encode_mv_i8(enc, wq, sq, &self.x_norm_buf, &self.q_buf, q_dim, hidden);
                    } else {
                        self.ops.encode_mv_f16(enc, wq, &self.x_norm_buf, &self.q_buf, q_dim, hidden);
                    }
                }
                if !is_kv_shared {
                    let wk = self.get_w(&format!("{pref}.self_attn.k_proj.weight"));
                    let wv = self.get_w(&format!("{pref}.self_attn.v_proj.weight"));
                    let k_dtype = self.try_get_dtype(&format!("{pref}.self_attn.k_proj.weight"))
                        .unwrap_or("f16");
                    if k_dtype == "i4" {
                        let sk = self.get_w(&format!("{pref}.self_attn.k_proj.weight.qscale"));
                        let sv = self.get_w(&format!("{pref}.self_attn.v_proj.weight.qscale"));
                        self.ops.encode_mv_i4(enc, wk, sk, &self.x_norm_buf, &self.k_buf, kv_dim, hidden, hidden);
                        self.ops.encode_mv_i4(enc, wv, sv, &self.x_norm_buf, &self.v_buf, kv_dim, hidden, hidden);
                    } else {
                        let sk = self.try_get_w(&format!("{pref}.self_attn.k_proj.weight.qscale"));
                        let sv = self.try_get_w(&format!("{pref}.self_attn.v_proj.weight.qscale"));
                        if let (Some(sk), Some(sv)) = (sk, sv) {
                            self.ops.encode_mv_i8(enc, wk, sk, &self.x_norm_buf, &self.k_buf, kv_dim, hidden);
                            self.ops.encode_mv_i8(enc, wv, sv, &self.x_norm_buf, &self.v_buf, kv_dim, hidden);
                        } else {
                            self.ops.encode_mv_f16(enc, wk, &self.x_norm_buf, &self.k_buf, kv_dim, hidden);
                            self.ops.encode_mv_f16(enc, wv, &self.x_norm_buf, &self.v_buf, kv_dim, hidden);
                        }
                    }
                }
            }

            // 4c. Per-head Q RMSNorm (in-place, weight shared across all Q heads).
            {
                let w_qn = self.get_w(&format!("{pref}.self_attn.q_norm.weight"));
                for h in 0..n_heads {
                    let byte_off = (h * head_dim * 4) as u64;
                    self.ops.encode_rms_norm_f16w_at(
                        enc,
                        &self.q_buf, byte_off,
                        w_qn, 0,
                        &self.q_buf, byte_off,
                        head_dim, cfg.rms_norm_eps, add_one,
                    );
                }
            }

            // 4d. Per-head K RMSNorm (in-place, skip for shared-KV layers).
            if !is_kv_shared {
                let w_kn = self.get_w(&format!("{pref}.self_attn.k_norm.weight"));
                for h in 0..n_kv_heads {
                    let byte_off = (h * kv_head_dim * 4) as u64;
                    self.ops.encode_rms_norm_f16w_at(
                        enc,
                        &self.k_buf, byte_off,
                        w_kn, 0,
                        &self.k_buf, byte_off,
                        kv_head_dim, cfg.rms_norm_eps, add_one,
                    );
                }
            }

            // 4d2. Gemma4 V-norm (per-head RMSNorm without weight, skip for shared-KV).
            // Can be disabled via env var CELLM_GEMMA4_DISABLE_VNORM=1 for debugging.
            let disable_vnorm = std::env::var("CELLM_GEMMA4_DISABLE_VNORM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if self.is_gemma4 && !is_kv_shared && !disable_vnorm {
                for h in 0..n_kv_heads {
                    let byte_off = (h * kv_head_dim * 4) as u64;
                    // RMSNorm with identity weight (all ones in f16 = 0x3C00)
                    self.ops.encode_rms_norm_f16w_at(
                        enc,
                        &self.v_buf, byte_off,
                        &self.v_norm_ones_buf, 0,
                        &self.v_buf, byte_off,
                        kv_head_dim, cfg.rms_norm_eps,
                        false, // add_one = false for Gemma4
                    );
                }
            }

            // 4e. rotate_half RoPE on Q (and K for non-shared layers).
            self.ops.encode_rope_half_f32(enc, &self.q_buf, n_heads,    head_dim,    head_dim,    pos, rope_theta);
            if !is_kv_shared {
                self.ops.encode_rope_half_f32(enc, &self.k_buf, n_kv_heads, kv_head_dim, kv_head_dim, pos, rope_theta);
            }

            // 4f. Write current token K/V to paged cache (skip for shared-KV layers).
            {
                let kv_store = kv_cache.storage().as_any()
                    .downcast_ref::<cellm_cache::kvcache::MetalKvStorage>()
                    .expect("GemmaGraphState requires MetalKvStorage backend");
                if !is_kv_shared {
                    let target_base = kv_cache.layout().token_base_elem(block_id, layer, token_off)?;
                    kv_store.encode_write_token_f32(enc, target_base, &self.k_buf, &self.v_buf, kv_dim);
                }

                // 4g. GQA attention (uses kv_src_layer's cache slot via bases_off).
                let attn_scale: Option<f32> = if self.is_gemma4 { Some(1.0) } else { None };
                kv_store.encode_attention(
                    enc,
                    bases_ref,
                    bases_off,
                    &self.q_buf,
                    &self.attn_out_buf,
                    attn_count,
                    n_heads    as u32,
                    n_kv_heads as u32,
                    head_dim   as u32,
                    attn_scale,
                    None, // soft_cap: Gemma3 does not use it
                );
            }

            // 4h. o_proj: attn_out → mlp_in_buf  (temporary storage for the attn branch).
            {
                let wo = self.get_w(&format!("{pref}.self_attn.o_proj.weight"));
                let o_dtype = self.try_get_dtype(&format!("{pref}.self_attn.o_proj.weight"))
                    .unwrap_or("f16");
                if o_dtype == "i4" {
                    let so = self.get_w(&format!("{pref}.self_attn.o_proj.weight.qscale"));
                    self.ops.encode_mv_i4(enc, wo, so, &self.attn_out_buf, &self.mlp_in_buf, hidden, q_dim, q_dim);
                } else {
                    let so = self.try_get_w(&format!("{pref}.self_attn.o_proj.weight.qscale"))
                        .or_else(|| self.try_get_w(&format!("{pref}.self_attn.o_proj.qscale")));
                    if let Some(so) = so {
                        self.ops.encode_mv_i8(enc, wo, so, &self.attn_out_buf, &self.mlp_in_buf, hidden, q_dim);
                    } else {
                        self.ops.encode_mv_f16(enc, wo, &self.attn_out_buf, &self.mlp_in_buf, hidden, q_dim);
                    }
                }
            }

            // 4i. Post-attention RMSNorm on attn branch: mlp_in → x_norm.
            {
                let w_pa = self.get_w(&format!("{pref}.post_attention_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.mlp_in_buf, w_pa, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4j. Residual add: x += post_attn_normed.
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.x_norm_buf, hidden);

            // 4k. Pre-FFN RMSNorm: x → mlp_in  (mlp_in now holds the FFN input).
            {
                let w_pf = self.get_w(&format!("{pref}.pre_feedforward_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_pf, &self.mlp_in_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4l. Gate + Up projections.
            {
                let wg = self.get_w(&format!("{pref}.mlp.gate_proj.weight"));
                let wu = self.get_w(&format!("{pref}.mlp.up_proj.weight"));
                let g_dtype = self.try_get_dtype(&format!("{pref}.mlp.gate_proj.weight"))
                    .unwrap_or("f16");
                if g_dtype == "i4" {
                    let sg = self.get_w(&format!("{pref}.mlp.gate_proj.weight.qscale"));
                    let su = self.get_w(&format!("{pref}.mlp.up_proj.weight.qscale"));
                    self.ops.encode_mv_i4(enc, wg, sg, &self.mlp_in_buf, &self.gate_buf, ffn_dim, hidden, hidden);
                    self.ops.encode_mv_i4(enc, wu, su, &self.mlp_in_buf, &self.up_buf,   ffn_dim, hidden, hidden);
                } else {
                    let sg = self.try_get_w(&format!("{pref}.mlp.gate_proj.weight.qscale"))
                        .or_else(|| self.try_get_w(&format!("{pref}.mlp.gate_proj.qscale")));
                    let su = self.try_get_w(&format!("{pref}.mlp.up_proj.weight.qscale"))
                        .or_else(|| self.try_get_w(&format!("{pref}.mlp.up_proj.qscale")));
                    if let (Some(sg), Some(su)) = (sg, su) {
                        self.ops.encode_mv_i8(enc, wg, sg, &self.mlp_in_buf, &self.gate_buf, ffn_dim, hidden);
                        self.ops.encode_mv_i8(enc, wu, su, &self.mlp_in_buf, &self.up_buf,   ffn_dim, hidden);
                    } else {
                        self.ops.encode_mv_f16(enc, wg, &self.mlp_in_buf, &self.gate_buf, ffn_dim, hidden);
                        self.ops.encode_mv_f16(enc, wu, &self.mlp_in_buf, &self.up_buf,   ffn_dim, hidden);
                    }
                }
            }

            // 4m. GELU activation: gate = gelu(gate) * up.
            self.ops.encode_gelu_tanh_mul_f32_inplace(enc, &self.gate_buf, &self.up_buf, ffn_dim);

            // 4n. Down projection.
            {
                let wd = self.get_w(&format!("{pref}.mlp.down_proj.weight"));
                let d_dtype = self.try_get_dtype(&format!("{pref}.mlp.down_proj.weight"))
                    .unwrap_or("f16");
                if d_dtype == "i4" {
                    let sd = self.get_w(&format!("{pref}.mlp.down_proj.weight.qscale"));
                    self.ops.encode_mv_i4(enc, wd, sd, &self.gate_buf, &self.down_buf, hidden, ffn_dim, ffn_dim);
                } else {
                    let sd = self.try_get_w(&format!("{pref}.mlp.down_proj.weight.qscale"))
                        .or_else(|| self.try_get_w(&format!("{pref}.mlp.down_proj.qscale")));
                    if let Some(sd) = sd {
                        self.ops.encode_mv_i8(enc, wd, sd, &self.gate_buf, &self.down_buf, hidden, ffn_dim);
                    } else {
                        self.ops.encode_mv_f16(enc, wd, &self.gate_buf, &self.down_buf, hidden, ffn_dim);
                    }
                }
            }

            // 4o. Post-FFN RMSNorm on MLP branch: down → x_norm.
            {
                let w_pff = self.get_w(&format!("{pref}.post_feedforward_layernorm.weight"));
                self.ops.encode_rms_norm_f16w(enc, &self.down_buf, w_pff, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
            }

            // 4p. Residual add: x += post_ffn_normed.
            self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.x_norm_buf, hidden);

            // 4q. Gemma4 PLE injection + layer output scaling (GPU path).
            let layer_has_ple = ple_data.as_ref()
                .map(|&(ref ple_in, ppl)| {
                    ppl > 0 && layer * ppl + ppl <= ple_in.len()
                })
                .unwrap_or(false);

            if layer_has_ple {
                let ppl = ple_data.as_ref().unwrap().1;

                // Copy current layer's PLE slice from ple_buf to ple_tmp_buf.
                if let (Some(ref ple_buf), Some(ref ple_tmp)) = (&self.ple_buf, &self.ple_tmp_buf) {
                    self.ops.encode_copy_f32(enc, ple_buf, ple_tmp, layer * ppl, 0, ppl);
                }

                // PLE gate: gate_buf = x_buf @ W_gate  (hidden → ppl)
                let gate_name = format!("{pref}.per_layer_input_gate.weight");
                let proj_name = format!("{pref}.per_layer_projection.weight");
                let norm_name = format!("{pref}.post_per_layer_input_norm.weight");

                if let (Some(ref ple_tmp), Some(w_gate), Some(w_proj), Some(w_norm)) = (
                    self.ple_tmp_buf.as_ref(),
                    self.try_get_w(&gate_name),
                    self.try_get_w(&proj_name),
                    self.try_get_w(&norm_name),
                ) {
                    // Determine dtype for gate weight
                    let gate_dtype = self.try_get_dtype(&gate_name).unwrap_or("f16");
                    if gate_dtype == "i4" {
                        let s_gate = self.get_w(&format!("{gate_name}.qscale"));
                        self.ops.encode_mv_i4(enc, w_gate, s_gate, &self.x_buf, &self.gate_buf, ppl, hidden, hidden);
                    } else {
                        let s_gate = self.try_get_w(&format!("{gate_name}.qscale"));
                        if let Some(sg) = s_gate {
                            self.ops.encode_mv_i8(enc, w_gate, sg, &self.x_buf, &self.gate_buf, ppl, hidden);
                        } else {
                            self.ops.encode_mv_f16(enc, w_gate, &self.x_buf, &self.gate_buf, ppl, hidden);
                        }
                    }

                    // gate_buf = GELU(gate_buf) * ple_tmp  (in-place on gate_buf)
                    self.ops.encode_gelu_tanh_mul_f32_inplace(enc, &self.gate_buf, ple_tmp, ppl);

                    // Determine dtype for proj weight
                    let proj_dtype = self.try_get_dtype(&proj_name).unwrap_or("f16");
                    if proj_dtype == "i4" {
                        let s_proj = self.get_w(&format!("{proj_name}.qscale"));
                        self.ops.encode_mv_i4(enc, w_proj, s_proj, &self.gate_buf, &self.mlp_in_buf, hidden, ppl, ppl);
                    } else {
                        let s_proj = self.try_get_w(&format!("{proj_name}.qscale"));
                        if let Some(sp) = s_proj {
                            self.ops.encode_mv_i8(enc, w_proj, sp, &self.gate_buf, &self.mlp_in_buf, hidden, ppl);
                        } else {
                            self.ops.encode_mv_f16(enc, w_proj, &self.gate_buf, &self.mlp_in_buf, hidden, ppl);
                        }
                    }

                    // x_norm_buf = RMSNorm(mlp_in_buf, W_norm)
                    self.ops.encode_rms_norm_f16w(enc, &self.mlp_in_buf, w_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, false);

                    // x_buf += x_norm_buf (residual add from PLE)
                    self.ops.encode_add_f32_inplace(enc, &self.x_buf, &self.x_norm_buf, hidden);

                    // Layer output scaling (Gemma4 per-layer scalar/vector multiply).
                    // The scale values are pre-loaded into layer_scale_buf at this
                    // layer's index before the encoder was created.
                    if let Some(ref ls_buf) = self.layer_scale_buf {
                        self.ops.encode_mul_f32_inplace(enc, &self.x_buf, ls_buf, hidden);
                    }
                }
            }

            enc.end_encoding();
        }

        // 5. Final norm + logits in a separate encoder.
        let enc = cb.new_compute_command_encoder();
        {
            let w_norm = self.get_w(&format!("{prefix}model.norm.weight"));
            self.ops.encode_rms_norm_f16w(enc, &self.x_buf, w_norm, &self.x_norm_buf, hidden, cfg.rms_norm_eps, add_one);
        }

        // 6. LM-head logits (optional).
        if return_logits {
            // Try multiple candidate names with prefix-aware lookup
            let candidates = [
                format!("{prefix}lm_head.weight"),
                "lm_head.weight".to_string(),
                format!("{prefix}model.embed_tokens.weight"),
                "model.embed_tokens.weight".to_string(),
            ];
            let (lm_name, wl) = candidates.iter()
                .find_map(|name| self.try_get_w(name).map(|buf| (name.clone(), buf)))
                .expect("GemmaGraph: lm_head / embed_tokens weight not found");
            let l_dtype = self.try_get_dtype(&lm_name).unwrap_or("f16");
            if l_dtype == "i4" {
                let sl = self.get_w(&format!("{lm_name}.qscale"));
                self.ops.encode_mv_i4(enc, wl, sl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden, hidden);
            } else {
                let sl = self.try_get_w(&format!("{lm_name}.qscale"));
                if let Some(sl) = sl {
                    self.ops.encode_mv_i8(enc, wl, sl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
                } else {
                    self.ops.encode_mv_f16(enc, wl, &self.x_norm_buf, &self.logits_buf, cfg.vocab_size, hidden);
                }
            }
        }

        // 7. GPU sync point.
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        if !return_logits {
            return Ok(None);
        }

        // 8. Read back logits and check for divergence.
        let mut logits = vec![0.0f32; cfg.vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.logits_buf.contents() as *const f32,
                logits.as_mut_ptr(),
                cfg.vocab_size,
            );
        }

        // Gemma4 final logit softcapping: tanh(x/30)*30
        let softcap = if self.is_gemma4 { Some(30.0f32) } else { None };
        if let Some(cap) = softcap {
            if cap > 0.0 {
                for v in logits.iter_mut() {
                    *v = cap * (*v / cap).tanh();
                }
            }
        }

        let nan_count = logits.iter().filter(|v| v.is_nan()).count();
        let inf_count = logits.iter().filter(|v| v.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            return Err(CoreError::Backend(format!(
                "GemmaGraphState: divergence at pos {pos} (NaN={nan_count} Inf={inf_count})"
            )));
        }

        Ok(Some(logits))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct GemmaGraphState;
