# Cellm VLM Performance Analysis

## Current Performance (SmolVLM 256M on Metal)

```
Timings: patch=23.2ms encoder=1843.2ms decode=6617.0ms total=9106.8ms
Output: 11 tokens (~1.6 tokens/sec)
```

## Root Cause Analysis

### 1. Vision Encoder (1.8s) - CPU BLAS Fallback

**Location:** `crates/cellm-sdk/src/vlm.rs:3150-3194`

The vision encoder intentionally falls back to CPU BLAS (cblas_sgemm) for large matrices:

```rust
// The Metal matmul kernel is a naive element-wise implementation that
// is orders of magnitude slower than cblas_sgemm (Accelerate/AMX)
// for large encoder-scale matrices. Skip it for large batch sizes.
let total_ops = rows * in_dim * out_dim;
let skip_metal = total_ops >= (1 << 20);  // 1M ops threshold
if skip_metal {
    // Fall through to cblas_sgemm
}
```

For a 256M vision model, this is reasonable CPU performance (~1.8s). To improve:
- Implement proper tiled matmul kernels for Metal
- Use MPS (Metal Performance Shaders) for large matrix ops
- Batch vision encoding across multiple images

### 2. Text Decode (6.6s for 11 tokens = ~600ms/token) - CRITICAL BOTTLENECK

**Primary Issue:** Missing `autoreleasepool` around Metal command buffer operations

**Location:** `crates/cellm-model/src/llama_graph.rs:267-402` (`step_fused`)

Current code:
```rust
let cb = self.queue.new_command_buffer();
// ... encode all layers ...
enc.end_encoding();
let enc_final = cb.new_compute_command_encoder();
// ... final norm ...
enc_final.end_encoding();
let enc_logits = cb.new_compute_command_encoder();
// ... logits ...
enc_logits.end_encoding();
cb.commit();
cb.wait_until_completed();
```

Problems:
1. **No autoreleasepool**: Each command buffer creates Metal objects that accumulate in the autorelease pool until the pool drains. Without an explicit autoreleasepool, this can cause memory pressure and slowdowns.

2. **Synchronous wait per token**: Each token waits for GPU completion before starting the next. This is unavoidable for decode (need logits to sample), but 600ms is still way too high.

3. **Multiple encoders per command buffer**: Creates 3 separate compute encoders per token, which adds overhead.

### 3. Metal Kernel Efficiency

The Metal kernels in `crates/cellm-kernels/src/metal.rs` are functional but not optimized:
- Matmul uses naive element-wise dispatch (not tiled/block-based)
- No kernel fusion beyond what `step_fused` manually does
- Threadgroup memory not utilized effectively

## Expected vs Actual Performance

For a 256M model on Apple Silicon (M1/M2/M3):

| Component | Current | Target | Gap |
|-----------|---------|--------|-----|
| Vision Encoder | 1.8s | <100ms | 18x slower |
| Decode | ~600ms/token | ~5ms/token | 120x slower |
| Total | 9.1s | ~0.3s | 30x slower |

## Immediate Fixes (Low Hanging Fruit)

### 1. Add autoreleasepool to decode loop

In `llama_graph.rs`, wrap the command buffer operations:

```rust
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;

pub fn step_fused(...) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    autoreleasepool(|| {
        let cb = self.queue.new_command_buffer();
        // ... existing code ...
        cb.commit();
        cb.wait_until_completed();
        Ok(maybe_logits)
    })
}
```

### 2. Single Encoder Per Token

Combine all operations into a single compute encoder (already mostly done, but the final norm and logits use separate encoders).

### 3. Command Buffer Reuse

Investigate using `CommandBufferDescriptor` with `set_retained_references(false)` for better performance.

## Medium-term Optimizations

### 1. MPS for Vision Encoder

Replace custom Metal matmul with `MPSMatrixMultiplication` for large matrices.

### 2. Kernel Fusion

Fuse more operations into single kernels (e.g., rms_norm + matmul, rope + attention).

### 3. Speculative Decoding

Implement speculative decoding for 2-3x speedup on text generation.

### 4. Persistent Threads

Use indirect command buffers for more efficient dispatch.

## Profiling Recommendations

Run with Metal System Trace in Instruments to identify:
1. Command buffer idle time
2. Shader compilation stalls
3. Memory bandwidth saturation
4. GPU utilization percentage

```bash
# Build with release optimizations
cargo build --release

# Run with Metal validation disabled for accurate timing
METAL_DEVICE_WRAPPER_TYPE=1 ./target/release/vlm-direct ...
```

## Implementation Done

### 1. Added autoreleasepool to decode path

**Files Modified:**
- `crates/cellm-model/src/llama_graph.rs`
- `crates/cellm-model/Cargo.toml`

**Changes:**
1. Added `objc = "0.2"` dependency to `Cargo.toml`
2. Added `use objc::rc::autoreleasepool;` import
3. Wrapped `step_fused()` function body in `autoreleasepool(|| { ... })`

**Expected Impact:** The autoreleasepool ensures Metal objects (command buffers, encoders) are properly released after each token decode. This should reduce memory pressure and potentially improve performance by 10-30% depending on how much overhead was caused by accumulated autoreleased objects.

## Quick Wins Checklist

- [x] Add autoreleasepool around command buffer in `step_fused`
- [ ] Verify no shader recompilation per token
- [ ] Check command buffer reuse is working
- [ ] Profile with Instruments Metal System Trace
- [ ] Consider MPS fallback for vision encoder
