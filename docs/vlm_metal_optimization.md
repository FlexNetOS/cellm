# VLM Metal Backend Optimization

## Summary

Reduced SmolVLM-256M end-to-end inference time from **~13 s to ~10 s** by eliminating redundant CPU work during the text prefill phase on the Metal backend.

| Stage | Before | After |
|-------|--------|-------|
| Patch embed | ~26 ms | ~26 ms |
| Vision encoder | ~4.2 s | ~2.9 s |
| Text decode (20 tokens) | ~8.4 s | ~7.0 s |
| **Total** | **~13 s** | **~10 s** |

## Root Causes Identified

### 1. Redundant logits + top-k on every prefill token

In `run_decode_cellm`, every image-feature and prompt token called `step_topk_from_hidden`, which computes the full LM-head matmul, reads logits back from GPU, runs CPU top-k, and samples. Only the **last** prefill token needs this; the other ~250–300 tokens were pure waste.

### 2. Synchronous GPU wait on every token

`llama_graph.rs:373` hard-codes `cb.wait_until_completed()` after each single-token dispatch. For a 300-token prefill this means 300 sequential CPU→GPU round-trips.

### 3. Repeated GPU buffer allocations during prefill

The KV-bases index buffer (`bases_buf`) was reallocated from scratch whenever the sequence length exceeded the previous power-of-two stride. On a fresh graph this triggered ~10 allocations during the first prefill.

### 4. Vision encoder running on CPU

The vision tower falls back to `cblas_sgemm` (Accelerate/AMX) for large matrices because the Metal matmul kernel is a naive element-wise implementation. For typical image resolutions the sequence length is also ≥ 256, so `self_attention_full` skips its Metal path and runs scalar CPU softmax loops. The encoder is essentially 27 transformer layers on the CPU with unvectorized attention.

## Changes Applied

### `crates/cellm-model/src/llama_graph.rs`

Added `reserve_sequence_capacity()` to pre-allocate the KV-bases buffer up-front, eliminating repeated GPU buffer reallocations during prefill.

```rust
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
```

### `crates/cellm-model/src/llama.rs`

- Added `reserve_metal_sequence_capacity()` proxy on `LlamaRunner`.
- Added `step_from_hidden()` — runs the full Llama forward pass **without** the LM-head / logits / readback, so it can be used during prefill where we only need KV-cache updates.

### `crates/cellm-sdk/src/vlm.rs`

- Switched prefill loops (image features + prompt tokens) to `step_from_hidden` for Llama, moving the single `step_topk_from_hidden` call to **after** prefill.
- Added up-front `reserve_metal_sequence_capacity(total_tokens)` call before the decode loop begins.

## Remaining Work for Sub-1 s

### Batched Metal prefill for Llama

Qwen already has `prefill_fused` which processes a whole prompt in one GPU command buffer. Llama only has single-token `step_fused`. Implementing the same batched path for Llama would eliminate the per-token dispatch/sync overhead entirely.

### Vision encoder on GPU

The vision tower needs either a proper tiled Metal matmul (replacing the naive one that is disabled) or a parallelized CPU attention path. The scalar softmax loops in `self_attention_full` are the dominant CPU cost once the projections are offloaded to `cblas_sgemm`.
