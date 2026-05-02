# Why Gemma4 on Metal produced only `<unused33>` tokens

*May 2, 2026*

## Symptom
After the prefill stage, every generated token was `<unused33>` (token ID 33). The model worked fine on CPU.

## The short answer
The per-layer Metal decode path was reading K and V cache values from a GPU buffer that had never been filled. Attention computed scores against garbage, and after 35 layers the hidden state collapsed to a single fixed attractor -- token 33.

## What happened step by step

### 1. The fused fast path is disabled for Gemma4
`step_topk_batched` runs the entire model in one GPU command buffer. It is disabled for Gemma4 models that have shared KV layers. The model therefore falls back to the slower per-layer decode path.

### 2. The per-layer path tries to reuse a "persistent" GPU KV cache
The per-layer loop calls `attention_and_proj_i4_persistent_kv`. This function is supposed to:
- upload the current query vector to the GPU
- read all historical K and V values from a persistent per-layer GPU buffer
- run attention and the output projection on the GPU
- read the result back

### 3. The persistent GPU KV cache was empty for all historical positions
The buffer is allocated once per layer with `ensure_layer_kv_cache`, but it is only written to in two places:
- `write_kv_token` (called from the CPU decode path after each token)
- `encode_scatter_f32` inside `step_topk_batched` (the fused fast path)

**Prefill never writes to it.** Prefill runs on the CPU and stores K/V in the paged CPU KV cache. The GPU persistent cache is untouched during prefill, so all positions before the first decode step contain zeros or uninitialized memory.

**Shared KV layers never write to their own layer slot.** In Gemma4, the last 20 layers share K/V projections with earlier layers. The per-layer GPU cache index is based on layer number. Only the source layers ever had `write_kv_token` called; the shared layers had empty buffers.

### 4. Attention with garbage K/V produces garbage hidden states
The attention kernel `attention_gqa_f32_fast` reads K and V as `device const float*`. When those values are zeros or random data, the attention scores and weighted sums are meaningless. This corrupts the residual stream at every layer.

### 5. After 35 layers the hidden state collapses to a fixed point
A meaningless hidden state, passed through 35 layers of linear transformations, norms, and activations, converges to a nearly constant vector. The final logits are therefore nearly identical at every step. Token 33 happens to be the argmax of that degenerate distribution.

## Additional bug found in the same path
The `attention_and_proj_i4_persistent_kv` function also had a second bug: the `encode_mv_i4` call for the output projection passed its arguments in the wrong order (attention output buffer as weights, weight buffer as scales, scale buffer as input). Even if the KV cache had been valid, the output projection would have produced garbage.

## Fixes applied
1. Changed the per-layer decode path to bypass the broken persistent-KV attention and use the gather-and-upload path instead (`gpu_attention_and_proj`). This uploads the full K/V sequence from the CPU paged cache to the GPU every step, so the attention kernel always sees valid data.
2. Fixed the swapped `encode_mv_i4` arguments in `attention_and_proj_i4_persistent_kv` for any future code that re-enables the persistent cache path.
3. Fixed `write_kv_token` to write `f32` instead of `f16`, because the persistent cache buffers are sized for `f32` and the attention kernel reads `float*`.

## Result
After the fix, the model generates coherent text on the Metal backend instead of repeating `<unused33>`.
