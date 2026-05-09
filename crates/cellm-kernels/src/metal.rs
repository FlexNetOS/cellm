// Author: Jeffrey Asante (https://jeffasante.github.io/)
#[cfg(any(target_os = "macos", target_os = "ios"))]
use metal::*;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::{msg_send, sel, sel_impl};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::rc::autoreleasepool;
use std::sync::Mutex;
use std::collections::HashMap;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::hash::{Hash, Hasher};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::hash_map::DefaultHasher;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::fs;

// Compiled Metal library cache — MSL is compiled once per process so iOS cold-launches
// don't pay the compile cost on every app restart after a cache eviction.
#[cfg(any(target_os = "macos", target_os = "ios"))]
static ELEM_OPS_LIB_CACHE: Mutex<Option<Library>> = Mutex::new(None);

// PSO cache to avoid recompiling shaders on every MetalOps::create()
#[cfg(any(target_os = "macos", target_os = "ios"))]
static PSO_CACHE: Mutex<Option<HashMap<String, ComputePipelineState>>> = Mutex::new(None);

/// Load a compiled .metallib from disk, or compile from source and cache it.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn load_or_compile_metallib(
    device: &Device,
    source: &str,
    fast_math: bool,
    cache_name: &str,
) -> anyhow::Result<Library> {
    // Hash source + options to form a versioned filename
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    fast_math.hash(&mut hasher);
    let hash = format!("{:016x}", hasher.finish());

    #[cfg(target_os = "ios")]
    let cache_dir = std::env::temp_dir().join("cellm_shaders");
    #[cfg(not(target_os = "ios"))]
    let cache_dir = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".cache/cellm/shaders"))
        .unwrap_or_else(|| std::env::temp_dir().join("cellm_shaders"));
    let cache_path = cache_dir.join(format!("{}_{}.metallib", cache_name, hash));

    // Try loading pre-compiled library
    if cache_path.exists() {
        match device.new_library_with_file(&cache_path) {
            Ok(lib) => return Ok(lib),
            Err(e) => {
                eprintln!("cellm: warning: failed to load cached metallib, recompiling: {}", e);
                let _ = fs::remove_file(&cache_path);
            }
        }
    }

    // Compile from source
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(fast_math);
    let lib = device
        .new_library_with_source(source, &options)
        .map_err(|e| anyhow::anyhow!("Compile failed: {e:?}"))?;

    // Serialize to disk
    if let Some(path_str) = cache_path.to_str() {
        if fs::create_dir_all(&cache_dir).is_err() {
            eprintln!("cellm: warning: failed to create metallib cache dir: {:?}", cache_dir);
        } else {
            let url_str = format!("file://{}", path_str);
            let url = metal::URL::new_with_string(&url_str);
            unsafe {
                use objc::runtime::Object;
                let mut err: *mut Object = std::ptr::null_mut();
                let _result: bool = msg_send![&*lib, serializeToURL: &*url error: &mut err];
                if !err.is_null() {
                    let desc: *mut Object = msg_send![err, localizedDescription];
                    let cstr: *const std::ffi::c_char = msg_send![desc, UTF8String];
                    let msg = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                    eprintln!("cellm: warning: failed to serialize metallib to cache: {}", msg);
                }
            }
        }
    } else {
        eprintln!("cellm: warning: metallib cache path is not valid UTF-8, skipping cache");
    }

    Ok(lib)
}

pub struct MetalKernels;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub type MetalBuffer = Buffer;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalBuffer;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalMatmul {
    pub queue: CommandQueue,
    pub _lib: Library,
    pub pso: ComputePipelineState,
    pub pso_vec4: ComputePipelineState,
    pub pso_tiled: ComputePipelineState,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct MetalMatmul;

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Metal: no device found (system_default and all() empty). \
                     If you are in a restricted/sandboxed shell, re-run outside sandbox."
                )
            })?;
        let queue = device.new_command_queue();

        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
        "#;

        let (lib, pso) = build_pipeline(&device, src, "add_f32")?;

        let n = 1024usize;
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let a = upload_f32(&device, &(0..n).map(|i| i as f32).collect::<Vec<_>>())?;
        let b = upload_f32(&device, &(0..n).map(|i| (2 * i) as f32).collect::<Vec<_>>())?;
        let out = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&a), 0);
        enc.set_buffer(1, Some(&b), 0);
        enc.set_buffer(2, Some(&out), 0);
        let w = pso.thread_execution_width() as u64;
        let tg = MTLSize { width: w.min(n as u64), height: 1, depth: 1 };
        let grid = MTLSize { width: n as u64, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Validate.
        let out_ptr = out.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        for i in 0..n {
            let expected = (i as f32) + (2 * i) as f32;
            let got = out_slice[i];
            if (got - expected).abs() > 1e-5 {
                anyhow::bail!("Metal add_f32 mismatch at {i}: got={got} expected={expected}");
            }
        }

        let _ = lib;
        Ok(())
    }

    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        let device = Device::system_default()
            .or_else(|| Device::all().into_iter().next())
            .ok_or_else(|| anyhow::anyhow!("Metal: no device found"))?;
        let queue = device.new_command_queue();
        let src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void matmul_f32(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& m [[buffer(3)]],
            constant uint& n [[buffer(4)]],
            constant uint& k [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;
            if (row < m && col < n) {
                float acc = 0.0f;
                for (uint kk = 0; kk < k; ++kk) {
                    acc += a[row * k + kk] * b[kk * n + col];
                }
                out[row * n + col] = acc;
            }
        }
        "#;
        let (lib, pso) = build_pipeline(&device, src, "matmul_f32")?;
        let src_vec4 = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void matmul_f32_vec4(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& m [[buffer(3)]],
            constant uint& n [[buffer(4)]],
            constant uint& k [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint row = gid.y;
            uint col = gid.x;
            if (row >= m || col * 4 >= n) return;
            float4 acc = float4(0.0f);
            for (uint kk = 0; kk < k; ++kk) {
                float a_val = a[row * k + kk];
                float4 b_val = *(device const float4*)(b + kk * n + col * 4);
                acc += a_val * b_val;
            }
            uint base = row * n + col * 4;
            out[base + 0] = acc.x;
            if (col * 4 + 1 < n) out[base + 1] = acc.y;
            if (col * 4 + 2 < n) out[base + 2] = acc.z;
            if (col * 4 + 3 < n) out[base + 3] = acc.w;
        }
        "#;
        let (_, pso_vec4) = build_pipeline(&device, src_vec4, "matmul_f32_vec4")?;
        let src_tiled = r#"
        #include <metal_stdlib>
        using namespace metal;

        // Tiled matmul: each threadgroup computes a 16x16 output tile.
        // Threadgroup memory caches 16x16 tiles of A and B per outer-loop
        // iteration, dramatically reducing global-memory reads vs the naive
        // kernel.
        kernel void matmul_tiled_f32(
            device const float* a [[buffer(0)]],  // M x K  row-major
            device const float* b [[buffer(1)]],  // K x N  row-major
            device       float* out [[buffer(2)]], // M x N  row-major
            constant     uint&  M     [[buffer(3)]],
            constant     uint&  N     [[buffer(4)]],
            constant     uint&  K     [[buffer(5)]],
            uint2 gid [[threadgroup_position_in_grid]],
            uint2 lid [[thread_position_in_threadgroup]],
            uint2 tsize[[threads_per_threadgroup]]
        ) {
            const uint TILE = 16;
            threadgroup float a_tile[TILE][TILE];
            threadgroup float b_tile[TILE][TILE];

            uint row_start = gid.y * TILE;
            uint col_start = gid.x * TILE;
            float acc = 0.0f;

            for (uint k = 0; k < K; k += TILE) {
                // Cooperative load 16x16 tile of A
                uint ar = row_start + lid.y;
                uint ac = k + lid.x;
                a_tile[lid.y][lid.x] = (ar < M && ac < K) ? a[ar * K + ac] : 0.0f;

                // Cooperative load 16x16 tile of B
                uint br = k + lid.y;
                uint bc = col_start + lid.x;
                b_tile[lid.y][lid.x] = (br < K && bc < N) ? b[br * N + bc] : 0.0f;

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Accumulate over the tile
                for (uint i = 0; i < TILE; ++i) {
                    acc += a_tile[lid.y][i] * b_tile[i][lid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            uint out_row = row_start + lid.y;
            uint out_col = col_start + lid.x;
            if (out_row < M && out_col < N) {
                out[out_row * N + out_col] = acc;
            }
        }
        "#;
        let (_, pso_tiled) = build_pipeline(&device, src_tiled, "matmul_tiled_f32")?;
        Ok(MetalMatmul {
            queue,
            _lib: lib,
            pso,
            pso_vec4,
            pso_tiled,
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalKernels {
    pub fn smoke_test_add_f32() -> anyhow::Result<()> {
        anyhow::bail!("MetalKernels only supported on macOS/iOS")
    }
    pub fn create_matmul() -> anyhow::Result<MetalMatmul> {
        anyhow::bail!("Metal matmul only supported on Apple platforms")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalMatmul {
    /// Like `matmul_row_major_f32_with_b_buffer` but uses a float4-vectorized
    /// kernel when `n` is divisible by 4, giving ~2-4x higher GPU ALU
    /// utilization on the vision-encoder-scale matrices that the scalar
    /// kernel struggles with.
        pub fn matmul_row_major_f32_fast(
            &self,
            a_buf: &MetalBuffer,
            m: usize,
            k: usize,
            b_buf: &MetalBuffer,
            n: usize,
            out: &mut [f32],
        ) -> anyhow::Result<()> {
            if out.len() != m * n {
                anyhow::bail!("Metal matmul output size mismatch");
            }
            let use_vec4 = n % 4 == 0;
            let pso = if use_vec4 { &self.pso_vec4 } else { &self.pso };
            autoreleasepool(|| {
                let out_bytes = (out.len() * 4) as u64;
                let device = self.queue.device();
                let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;

                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(a_buf), 0);
                enc.set_buffer(1, Some(b_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, 4, (&m_u32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n_u32 as *const u32).cast());
                enc.set_bytes(5, 4, (&k_u32 as *const u32).cast());

                let w = pso.thread_execution_width() as u64;
                let h = pso.max_total_threads_per_threadgroup() as u64 / w;
                let tg = MTLSize { width: w, height: h, depth: 1 };
                let grid = if use_vec4 {
                    MTLSize { width: (n / 4) as u64, height: m as u64, depth: 1 }
                } else {
                    MTLSize { width: n as u64, height: m as u64, depth: 1 }
                };
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                let out_ptr = out_buf.contents() as *const f32;
                out.copy_from_slice(unsafe { std::slice::from_raw_parts(out_ptr, out.len()) });
                Ok(())
            })
        }

        /// Tiled matmul using threadgroup memory for cache reuse.
        /// Uses 16x16 tiles with a 16x16 threadgroup (256 threads).
        /// Significantly faster than the naive element-wise kernel for matrices
        /// larger than ~[64, 64] because each element of A and B is read from
        /// global memory only K/TILE times instead of K times per output element.
        pub fn matmul_row_major_f32_tiled(
            &self,
            a_buf: &MetalBuffer,
            m: usize,
            k: usize,
            b_buf: &MetalBuffer,
            n: usize,
            out: &mut [f32],
        ) -> anyhow::Result<()> {
            if out.len() != m * n {
                anyhow::bail!("Metal tiled matmul output size mismatch");
            }
            autoreleasepool(|| {
                let out_bytes = (out.len() * 4) as u64;
                let device = self.queue.device();
                let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;

                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pso_tiled);
                enc.set_buffer(0, Some(a_buf), 0);
                enc.set_buffer(1, Some(b_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, 4, (&m_u32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n_u32 as *const u32).cast());
                enc.set_bytes(5, 4, (&k_u32 as *const u32).cast());

                // Dispatch 16x16 threadgroups
                let tg = MTLSize { width: 16, height: 16, depth: 1 };
                // ceil division for grid
                let grid_w = (n + 15) / 16;
                let grid_h = (m + 15) / 16;
                let grid = MTLSize { width: grid_w as u64, height: grid_h as u64, depth: 1 };
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                let out_ptr = out_buf.contents() as *const f32;
                out.copy_from_slice(unsafe { std::slice::from_raw_parts(out_ptr, out.len()) });
                Ok(())
            })
        }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalMatmul {
    pub fn upload_f32(&self, src: &[f32]) -> anyhow::Result<MetalBuffer> {
        upload_f32(self.queue.device(), src)
    }

    pub fn matmul_row_major_f32(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            anyhow::bail!("Metal matmul shape mismatch");
        }
        autoreleasepool(|| {
            let device = self.queue.device();
            let a_buf = upload_f32(device, a)?;
            let b_buf = upload_f32(device, b)?;
            self.matmul_row_major_f32_with_b_buffer(&a_buf, m, k, &b_buf, n, out)
        })
    }

    pub fn matmul_row_major_f32_with_b_buffer(
        &self,
        a_buf: &MetalBuffer,
        m: usize,
        k: usize,
        b_buf: &MetalBuffer,
        n: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        if out.len() != m * n { anyhow::bail!("Metal matmul output size mismatch"); }
        autoreleasepool(|| {
            let out_bytes = (out.len() * 4) as u64;
            let device = self.queue.device();
            let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

            let m_u32 = m as u32; let n_u32 = n as u32; let k_u32 = k as u32;

            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pso);
            enc.set_buffer(0, Some(a_buf), 0);
            enc.set_buffer(1, Some(b_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, (&m_u32 as *const u32).cast());
            enc.set_bytes(4, 4, (&n_u32 as *const u32).cast());
            enc.set_bytes(5, 4, (&k_u32 as *const u32).cast());

            let w = self.pso.thread_execution_width() as u64;
            let h = self.pso.max_total_threads_per_threadgroup() as u64 / w;
            let tg = MTLSize { width: w, height: h, depth: 1 };
            let grid = MTLSize { width: n as u64, height: m as u64, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let out_ptr = out_buf.contents() as *const f32;
            out.copy_from_slice(unsafe { std::slice::from_raw_parts(out_ptr, out.len()) });
            Ok(())
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalMatmul {
    pub fn upload_f32(&self, _: &[f32]) -> anyhow::Result<MetalBuffer> { anyhow::bail!("No Metal") }
    pub fn matmul_row_major_f32(&self, _: &[f32], _: usize, _: usize, _: &[f32], _: usize, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No Metal") }
    pub fn matmul_row_major_f32_with_b_buffer(&self, _: &MetalBuffer, _: usize, _: usize, _: &MetalBuffer, _: usize, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No Metal") }
    pub fn matmul_row_major_f32_fast(&self, _: &MetalBuffer, _: usize, _: usize, _: &MetalBuffer, _: usize, _: &mut [f32]) -> anyhow::Result<()> { anyhow::bail!("No Metal") }
}

const ELEM_OPS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) a[gid] += b[gid];
}

kernel void mul_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) a[gid] *= b[gid];
}

kernel void silu_mul_f32_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float x = a[gid];
        float silu = x / (1.0f + exp(-x));
        a[gid] = silu * b[gid];
    }
}

kernel void gelu_tanh_mul_f32_inplace(
    device float* gate [[buffer(0)]],
    device const float* up  [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float x = gate[gid];
    // tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    float c = 0.7978845608f;
    float v = c * (x + 0.044715f * x * x * x);
    gate[gid] = 0.5f * x * (1.0f + tanh(v)) * up[gid];
}

kernel void rms_norm_f16w(
    device const float*  x         [[buffer(10)]],
    device const half*   w         [[buffer(1)]],
    device       float*  out       [[buffer(2)]],
    constant     uint&   n         [[buffer(3)]],
    constant     float&  eps       [[buffer(4)]],
    constant     uint&   w_add_one [[buffer(5)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgsize[[threads_per_threadgroup]],
    threadgroup float* shared[[threadgroup(0)]]
) {
    // Phase 1: each thread computes partial sum of squares
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tgsize) {
        float v = x[i];
        partial += v * v;
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree reduction in shared memory
    for (uint s = tgsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: thread 0 computes inv_rms
    if (tid == 0) {
        float mean_sq = shared[0] / float(n);
        shared[0] = rsqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared[0];

    // Phase 4: apply normalization with weight
    for (uint i = tid; i < n; i += tgsize) {
        float wi = float(w[i]);
        if (w_add_one) wi += 1.0f;
        out[i] = x[i] * inv_rms * wi;
    }
}

kernel void rope_adj_f32(
    device       float* x         [[buffer(0)]],
    constant     uint&  n_heads   [[buffer(1)]],
    constant     uint&  head_dim  [[buffer(2)]],
    constant     uint&  pos       [[buffer(3)]],
    constant     float& theta     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_hd = head_dim >> 1;
    uint h = gid / half_hd;
    uint dim = gid % half_hd;
    if (h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(dim)) / float(head_dim));
    float angle = float(pos) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = h * head_dim + dim * 2;
    float x0 = x[base]; float x1 = x[base + 1];
    x[base]     = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

kernel void rope_half_f32(
    device       float* x           [[buffer(0)]],
    constant     uint&  n_heads     [[buffer(1)]],
    constant     uint&  head_dim    [[buffer(2)]],
    constant     uint&  rotary_dim  [[buffer(3)]],
    constant     uint&  pos         [[buffer(4)]],
    constant     float& theta       [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_rd = rotary_dim >> 1;
    uint h = gid / half_rd;
    uint i = gid % half_rd;
    if (h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(i)) / float(rotary_dim));
    float angle = float(pos) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = h * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + half_rd + i];
    x[base + i]           = x0 * c - x1 * s;
    x[base + half_rd + i] = x1 * c + x0 * s;
}

kernel void mv_f16(
    device const ushort* A       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device const float*  bias    [[buffer(2)]],
    device       float*  out     [[buffer(3)]],
    constant     uint&   rows    [[buffer(4)]],
    constant     uint&   cols    [[buffer(5)]],
    constant     uint&   has_bias [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const ushort* row_ptr = A + gid * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
    if (has_bias) acc += bias[gid];
    out[gid] = acc;
}

kernel void mv_i8(
    device const char*   A       [[buffer(0)]],
    device const ushort* scl     [[buffer(1)]],
    device const float*  x       [[buffer(2)]],
    device const float*  bias    [[buffer(3)]],
    device       float*  out     [[buffer(4)]],
    constant     uint&   rows    [[buffer(5)]],
    constant     uint&   cols    [[buffer(6)]],
    constant     uint&   has_bias [[buffer(7)]],
    constant     uint&   per_channel [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    float scale = float(as_type<half>(scl[per_channel ? gid : 0]));
    device const char* row_ptr = A + gid * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
    float res = acc * scale;
    if (has_bias) res += bias[gid];
    out[gid] = res;
}

kernel void mv_qkv_i8(
    device const char*   w_q      [[buffer(0)]],
    device const ushort* s_q      [[buffer(1)]],
    device const char*   w_k      [[buffer(2)]],
    device const ushort* s_k      [[buffer(3)]],
    device const char*   w_v      [[buffer(4)]],
    device const ushort* s_v      [[buffer(5)]],
    device const float*  x        [[buffer(6)]],
    device const float*  b_q      [[buffer(7)]],
    device const float*  b_k      [[buffer(8)]],
    device const float*  b_v      [[buffer(9)]],
    device       float*  q_out    [[buffer(10)]],
    device       float*  k_out    [[buffer(11)]],
    device       float*  v_out    [[buffer(12)]],
    constant     uint&   rows_q   [[buffer(13)]],
    constant     uint&   rows_kv  [[buffer(14)]],
    constant     uint&   cols     [[buffer(15)]],
    constant     uint&   has_bias [[buffer(16)]],
    constant     uint*   per_chan [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows_q + rows_kv + rows_kv;
    if (gid >= total) return;
    if (gid < rows_q) {
        float scale = float(as_type<half>(s_q[per_chan[0] ? gid : 0]));
        device const char* row_ptr = w_q + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
        float res = acc * scale;
        if (has_bias) res += b_q[gid];
        q_out[gid] = res;
        return;
    }
    uint off = gid - rows_q;
    if (off < rows_kv) {
        float scale = float(as_type<half>(s_k[per_chan[1] ? off : 0]));
        device const char* row_ptr = w_k + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
        float res = acc * scale;
        if (has_bias) res += b_k[off];
        k_out[off] = res;
        return;
    }
    off -= rows_kv;
    float scale = float(as_type<half>(s_v[per_chan[2] ? off : 0]));
    device const char* row_ptr = w_v + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(row_ptr[i]);
    float res = acc * scale;
    if (has_bias) res += b_v[off];
    v_out[off] = res;
}

kernel void mv_qkv_f16(
    device const ushort* w_q      [[buffer(0)]],
    device const ushort* w_k      [[buffer(1)]],
    device const ushort* w_v      [[buffer(2)]],
    device const float*  x        [[buffer(3)]],
    device const float*  b_q      [[buffer(4)]],
    device const float*  b_k      [[buffer(5)]],
    device const float*  b_v      [[buffer(6)]],
    device       float*  q_out    [[buffer(7)]],
    device       float*  k_out    [[buffer(8)]],
    device       float*  v_out    [[buffer(9)]],
    constant     uint&   rows_q   [[buffer(10)]],
    constant     uint&   rows_kv  [[buffer(11)]],
    constant     uint&   cols     [[buffer(12)]],
    constant     uint&   has_bias [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows_q + rows_kv + rows_kv;
    if (gid >= total) return;
    if (gid < rows_q) {
        device const ushort* row_ptr = w_q + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        if (has_bias) acc += b_q[gid];
        q_out[gid] = acc;
        return;
    }
    uint off = gid - rows_q;
    if (off < rows_kv) {
        device const ushort* row_ptr = w_k + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        if (has_bias) acc += b_k[off];
        k_out[off] = acc;
        return;
    }
    off -= rows_kv;
    device const ushort* row_ptr = w_v + off * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
    if (has_bias) acc += b_v[off];
    v_out[off] = acc;
}

kernel void mv2_f16(
    device const ushort* A0      [[buffer(0)]],
    device const ushort* A1      [[buffer(1)]],
    device const float*  x       [[buffer(2)]],
    device       float*  out0    [[buffer(3)]],
    device       float*  out1    [[buffer(4)]],
    constant     uint&   rows0   [[buffer(5)]],
    constant     uint&   rows1   [[buffer(6)]],
    constant     uint&   cols    [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows0 + rows1;
    if (gid >= total) return;
    if (gid < rows0) {
        device const ushort* row_ptr = A0 + gid * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        out0[gid] = acc;
    } else {
        uint off = gid - rows0;
        device const ushort* row_ptr = A1 + off * cols;
        float acc = 0.0f;
        for (uint i = 0; i < cols; ++i) acc += x[i] * float(as_type<half>(row_ptr[i]));
        out1[off] = acc;
    }
}

kernel void mv2_i8(
    device const char*   w0   [[buffer(0)]],
    device const char*   w1   [[buffer(1)]],
    device const ushort* s0   [[buffer(2)]],
    device const ushort* s1   [[buffer(3)]],
    device const float*  x    [[buffer(4)]],
    device       float*  o0   [[buffer(5)]],
    device       float*  o1   [[buffer(6)]],
    constant     uint&   r0   [[buffer(7)]],
    constant     uint&   r1   [[buffer(8)]],
    constant     uint&   c    [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = r0 + r1;
    if (gid >= total) return;
    if (gid < r0) {
        float scale = float(as_type<half>(s0[gid]));
        device const char* row_ptr = w0 + gid * c;
        float acc = 0.0f;
        for (uint i = 0; i < c; ++i) acc += x[i] * float(row_ptr[i]);
        o0[gid] = acc * scale;
    } else {
        uint off = gid - r0;
        float scale = float(as_type<half>(s1[off]));
        device const char* row_ptr = w1 + off * c;
        float acc = 0.0f;
        for (uint i = 0; i < c; ++i) acc += x[i] * float(row_ptr[i]);
        o1[off] = acc * scale;
    }
}
kernel void mv_q1_0_g128(
    device const uchar*  w     [[buffer(0)]],
    device const float*  x     [[buffer(1)]],
    device       float*  out   [[buffer(2)]],
    constant     uint&   rows  [[buffer(3)]],
    constant     uint&   cols  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const uchar* row_ptr = w + (ulong)gid * (cols / 128 * 18);
    float acc = 0.0f;
    for (uint i = 0; i < cols; i += 128) {
        uchar d_low = row_ptr[0];
        uchar d_high = row_ptr[1];
        ushort d_bits = ((ushort)d_high << 8) | (ushort)d_low;
        float d = (float)as_type<half>(d_bits);

        device const uchar* bits = row_ptr + 2;
        for (uint b = 0; b < 16; ++b) {
            uchar bb = bits[b];
            for (uint j = 0; j < 8; ++j) {
                float bit_val = (bb & (1 << j)) ? d : -d;
                acc += bit_val * x[i + b * 8 + j];
            }
        }
        row_ptr += 18;
    }
    out[gid] = acc;
}

kernel void lfm_conv(
    device       float*  state      [[buffer(0)]],  // [kernel_size, hidden]
    device const float*  input      [[buffer(1)]],  // [hidden]
    device const half*   weight     [[buffer(2)]],  // [hidden, kernel_size]
    device       float*  out        [[buffer(3)]],  // [hidden]
    constant     uint&   ks         [[buffer(4)]],
    constant     uint&   hidden     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden) return;
    // Shift state: slot i <- slot i+1, then write new input into last slot
    for (uint s = 0; s < ks - 1; ++s) {
        state[s * hidden + gid] = state[(s + 1) * hidden + gid];
    }
    state[(ks - 1) * hidden + gid] = input[gid];
    // Depthwise causal conv: dot product over kernel_size
    float acc = 0.0f;
    for (uint k = 0; k < ks; ++k) {
        acc += state[k * hidden + gid] * float(weight[gid * ks + k]);
    }
    out[gid] = acc;
}

kernel void mv_i4(
    device const uchar* w [[buffer(0)]],
    device const half* s [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* o [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& gs [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    uint spr = (gs > 0 && gs < cols) ? (cols / gs) : 1;
    device const half* rs = s + gid * spr;
    device const uchar* row_ptr = w + gid * (cols / 2);
    float acc = 0.0f;
    for (uint i = 0; i < cols; i += 2) {
        uchar packed = row_ptr[i / 2];
        float v0 = float(int(packed & 0x0F) - 8);
        float v1 = float(int(packed >> 4) - 8);
        float s0 = 1.0f;
        float s1 = 1.0f;
        if (gs > 0 && gs < cols) {
            s0 = float(rs[i / gs]);
            s1 = float(rs[(i + 1) / gs]);
        } else {
            s0 = float(rs[0]);
            s1 = float(rs[0]);
        }
        acc += v0 * x[i] * s0 + v1 * x[i + 1] * s1;
    }
    o[gid] = acc;
}

kernel void mv_f32(
    device const float* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* o [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const float* row_ptr = w + gid * cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) acc += x[i] * row_ptr[i];
    o[gid] = acc;
}

kernel void attention_gqa_f32(
    device const float* q         [[buffer(0)]],
    device const float* k         [[buffer(1)]],
    device const float* v         [[buffer(2)]],
    device       float* out       [[buffer(3)]],
    constant     uint&  n_heads   [[buffer(4)]],
    constant     uint&  n_kv_heads[[buffer(5)]],
    constant     uint&  head_dim  [[buffer(6)]],
    constant     uint&  seq_len   [[buffer(7)]],
    constant     float& scale     [[buffer(8)]],
    constant     float& soft_cap  [[buffer(9)]],
    constant     uint&  kv_stride [[buffer(10)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= n_heads) return;

    uint group_size = max(n_heads / n_kv_heads, 1u);
    uint kv_h = gid / group_size;

    device const float* qh = q + gid * head_dim;
    device       float* oh = out + gid * head_dim;

    for (uint i = 0; i < head_dim; i++) {
        oh[i] = 0.0f;
    }

    if (seq_len == 0) return;

    // Compute max score for numerical stability
    float max_score = -INFINITY;
    for (uint t = 0; t < seq_len; t++) {
        device const float* kt = k + t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; i++) {
            dot += qh[i] * kt[i];
        }
        float score = dot * scale;
        if (soft_cap > 0.0f) {
            score = tanh(score / soft_cap) * soft_cap;
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    // Compute softmax weights and weighted sum of V
    float sum = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        device const float* kt = k + t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; i++) {
            dot += qh[i] * kt[i];
        }
        float score = dot * scale;
        if (soft_cap > 0.0f) {
            score = tanh(score / soft_cap) * soft_cap;
        }
        float w = exp(score - max_score);
        sum += w;

        device const float* vt = v + t * kv_stride + kv_h * head_dim;
        for (uint i = 0; i < head_dim; i++) {
            oh[i] += w * vt[i];
        }
    }

    // Normalize
    for (uint i = 0; i < head_dim; i++) {
        oh[i] /= sum;
    }
}

// Threadgroup-parallel decode attention: 128 threads per head cooperate
// to parallelize over sequence tokens.  Uses shared-memory reductions
// for softmax and for accumulating weighted V sums dimension-by-dimension.
kernel void attention_gqa_f32_fast(
    device const float* q         [[buffer(0)]],
    device const float* k         [[buffer(1)]],
    device const float* v         [[buffer(2)]],
    device       float* out       [[buffer(3)]],
    constant     uint&  n_heads   [[buffer(4)]],
    constant     uint&  n_kv_heads[[buffer(5)]],
    constant     uint&  head_dim  [[buffer(6)]],
    constant     uint&  seq_len   [[buffer(7)]],
    constant     float& scale     [[buffer(8)]],
    constant     float& soft_cap  [[buffer(9)]],
    constant     uint&  kv_stride [[buffer(10)]],
    uint         tid    [[thread_index_in_threadgroup]],
    uint         gid    [[threadgroup_position_in_grid]],
    uint         tgsize [[threads_per_threadgroup]]
) {
    uint head = gid;
    uint group_size = max(n_heads / n_kv_heads, 1u);
    uint kv_h = head / group_size;

    uint n_threads = tgsize;
    uint thread_id = tid;

    device const float* qh = q + head * head_dim;
    device       float* oh = out + head * head_dim;

    if (seq_len == 0) return;

    // Divide sequence across threads in the group.
    uint tokens_per_thread = (seq_len + n_threads - 1) / n_threads;
    uint start_t = thread_id * tokens_per_thread;
    uint end_t   = min(start_t + tokens_per_thread, seq_len);

    // ---- Phase 1: local max score ----
    float local_max = -INFINITY;
    for (uint t = start_t; t < end_t; t++) {
        device const float* kt = k + t * kv_stride + kv_h * head_dim;
        float dp = 0.0f;
        uint i = 0;
        for (; i + 3 < head_dim; i += 4) {
            float4 qv = float4(qh[i], qh[i+1], qh[i+2], qh[i+3]);
            float4 kv = float4(kt[i], kt[i+1], kt[i+2], kt[i+3]);
            dp += dot(qv, kv);
        }
        for (; i < head_dim; i++) {
            dp += qh[i] * kt[i];
        }
        float score = dp * scale;
        if (soft_cap > 0.0f) {
            score = tanh(score / soft_cap) * soft_cap;
        }
        local_max = max(local_max, score);
    }

    // Reduce max across threadgroup
    threadgroup float shared[128];
    shared[thread_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id == 0) {
        float m = shared[0];
        for (uint i = 1; i < n_threads; i++) {
            m = max(m, shared[i]);
        }
        shared[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = shared[0];

    // ---- Phase 2: local sum of exponentials ----
    float local_sum = 0.0f;
    for (uint t = start_t; t < end_t; t++) {
        device const float* kt = k + t * kv_stride + kv_h * head_dim;
        float dp = 0.0f;
        uint i = 0;
        for (; i + 3 < head_dim; i += 4) {
            float4 qv = float4(qh[i], qh[i+1], qh[i+2], qh[i+3]);
            float4 kv = float4(kt[i], kt[i+1], kt[i+2], kt[i+3]);
            dp += dot(qv, kv);
        }
        for (; i < head_dim; i++) {
            dp += qh[i] * kt[i];
        }
        float score = dp * scale;
        if (soft_cap > 0.0f) {
            score = tanh(score / soft_cap) * soft_cap;
        }
        local_sum += exp(score - global_max);
    }

    // Reduce sum across threadgroup
    shared[thread_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id == 0) {
        float s = 0.0f;
        for (uint i = 0; i < n_threads; i++) {
            s += shared[i];
        }
        shared[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = shared[0];

    // ---- Phase 3: weighted V sum, reduced per dimension ----
    // We tile over head_dim to keep shared memory small (TILE * n_threads).
    const uint TILE = 32;
    threadgroup float tg_partial[TILE][128];

    for (uint d_base = 0; d_base < head_dim; d_base += TILE) {
        uint d_end = min(d_base + TILE, head_dim);
        uint tile_count = d_end - d_base;

        // Local partials for this tile
        float local_partials[TILE];
        for (uint i = 0; i < tile_count; i++) {
            local_partials[i] = 0.0f;
        }

        for (uint t = start_t; t < end_t; t++) {
            // Recompute weight for this token
            device const float* kt = k + t * kv_stride + kv_h * head_dim;
            float dp = 0.0f;
            uint i = 0;
            for (; i + 3 < head_dim; i += 4) {
                float4 qv = float4(qh[i], qh[i+1], qh[i+2], qh[i+3]);
                float4 kv = float4(kt[i], kt[i+1], kt[i+2], kt[i+3]);
                dp += dot(qv, kv);
            }
            for (; i < head_dim; i++) {
                dp += qh[i] * kt[i];
            }
            float score = dp * scale;
            if (soft_cap > 0.0f) {
                score = tanh(score / soft_cap) * soft_cap;
            }
            float w = exp(score - global_max);

            device const float* vt = v + t * kv_stride + kv_h * head_dim;
            for (uint i = 0; i < tile_count; i++) {
                local_partials[i] += w * vt[d_base + i];
            }
        }

        // Store partials in threadgroup memory
        for (uint i = 0; i < tile_count; i++) {
            tg_partial[i][thread_id] = local_partials[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 reduces and normalizes this tile
        if (thread_id == 0) {
            for (uint i = 0; i < tile_count; i++) {
                float acc = 0.0f;
                for (uint j = 0; j < n_threads; j++) {
                    acc += tg_partial[i][j];
                }
                oh[d_base + i] = acc / global_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void scatter_f32(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant     uint&  offset [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        dst[offset + gid] = src[gid];
    }
}

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant     uint&  src_offset [[buffer(2)]],
    constant     uint&  dst_offset [[buffer(3)]],
    constant     uint&  n           [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        dst[dst_offset + gid] = src[src_offset + gid];
    }
}

kernel void batch_mv_f16(
    device const ushort* A       [[buffer(0)]],  // [out_dim, in_dim] f16 weights, row-major
    device const float*  x_all   [[buffer(1)]],  // [num_tokens, in_dim] f32 input
    device       float*  out_all [[buffer(2)]],  // [num_tokens, out_dim] f32 output
    constant     uint&   num_tokens [[buffer(3)]],
    constant     uint&   out_dim    [[buffer(4)]],
    constant     uint&   in_dim     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint row = gid.x;
    if (token_idx >= num_tokens || row >= out_dim) return;

    device const ushort* row_ptr = A + row * in_dim;
    device const float* x_ptr = x_all + token_idx * in_dim;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; ++i) {
        acc += x_ptr[i] * float(as_type<half>(row_ptr[i]));
    }
    out_all[token_idx * out_dim + row] = acc;
}

kernel void batch_mv_i8(
    device const char*   A          [[buffer(0)]],
    device const ushort* scales     [[buffer(1)]],
    device const float*  x_all      [[buffer(2)]],
    device       float*  out_all    [[buffer(3)]],
    constant     uint&   num_tokens [[buffer(4)]],
    constant     uint&   out_dim    [[buffer(5)]],
    constant     uint&   in_dim     [[buffer(6)]],
    constant     uint&   per_chan   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint row = gid.x;
    if (token_idx >= num_tokens || row >= out_dim) return;

    float scale = float(as_type<half>(scales[per_chan ? row : 0]));
    device const char* row_ptr = A + row * in_dim;
    device const float* x_ptr = x_all + token_idx * in_dim;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; ++i) {
        acc += x_ptr[i] * float(row_ptr[i]);
    }
    out_all[token_idx * out_dim + row] = acc * scale;
}

// Batch RoPE for half-split (non-interleaved): applies RoPE to multi-token buffer in-place.
// Dispatch: 2D grid (num_pairs_per_token, num_tokens) where num_pairs = n_heads * head_dim/2
kernel void batch_rope_half_f32(
    device       float* x         [[buffer(0)]],  // [num_tokens, n_heads * head_dim]
    constant     uint&  num_tokens [[buffer(1)]],
    constant     uint&  n_heads   [[buffer(2)]],
    constant     uint&  head_dim  [[buffer(3)]],
    constant     uint&  rotary_dim [[buffer(4)]],
    constant     uint&  start_pos [[buffer(5)]],
    constant     float& theta     [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint pair_idx = gid.x;
    uint half_rd = rotary_dim >> 1;
    uint h = pair_idx / half_rd;
    uint i = pair_idx % half_rd;
    if (token_idx >= num_tokens || h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(i)) / float(rotary_dim));
    float angle = float(start_pos + token_idx) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = token_idx * n_heads * head_dim + h * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + half_rd + i];
    x[base + i]           = x0 * c - x1 * s;
    x[base + half_rd + i] = x1 * c + x0 * s;
}

// Batch RoPE for adjacent-pair (interleaved): applies RoPE to multi-token buffer in-place.
// Dispatch: 2D grid (num_pairs_per_token, num_tokens) where num_pairs = n_heads * head_dim/2
kernel void batch_rope_adj_f32(
    device       float* x         [[buffer(0)]],  // [num_tokens, n_heads * head_dim]
    constant     uint&  num_tokens [[buffer(1)]],
    constant     uint&  n_heads   [[buffer(2)]],
    constant     uint&  head_dim  [[buffer(3)]],
    constant     uint&  start_pos [[buffer(4)]],
    constant     float& theta     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint pair_idx = gid.x;
    uint half_hd = head_dim >> 1;
    uint h = pair_idx / half_hd;
    uint dim = pair_idx % half_hd;
    if (token_idx >= num_tokens || h >= n_heads) return;
    float inv_freq = pow(theta, -(2.0f * float(dim)) / float(head_dim));
    float angle = float(start_pos + token_idx) * inv_freq;
    float c = cos(angle); float s = sin(angle);
    uint base = token_idx * n_heads * head_dim + h * head_dim + dim * 2;
    float x0 = x[base]; float x1 = x[base + 1];
    x[base]     = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

// Batch write K/V: writes all tokens' K/V from multi-token buffers to KV cache.
// Dispatch: 2D grid (kv_dim, num_tokens)
// bases[pos] = base element in KV cache for position pos, layer-relative
kernel void batch_write_kv_f32(
    device half*   k_cache   [[buffer(0)]],
    device half*   v_cache   [[buffer(1)]],
    device const float* k_all [[buffer(2)]],  // [num_tokens, kv_dim]
    device const float* v_all [[buffer(3)]],  // [num_tokens, kv_dim]
    device const uint*  bases [[buffer(4)]],  // [stride] base per position (layer-relative)
    constant     uint&  num_tokens [[buffer(5)]],
    constant     uint&  kv_dim     [[buffer(6)]],
    constant     uint&  start_pos  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint d = gid.x;
    if (token_idx >= num_tokens || d >= kv_dim) return;
    uint pos = start_pos + token_idx;
    uint base = bases[pos];
    uint idx = base + d;
    k_cache[idx] = half(clamp(k_all[token_idx * kv_dim + d], -65504.0f, 65504.0f));
    v_cache[idx] = half(clamp(v_all[token_idx * kv_dim + d], -65504.0f, 65504.0f));
}

// Batched RMS norm: processes all tokens in one dispatch.
// Each threadgroup handles one token's RMS norm reduction.
// Dispatch grid: num_tokens threadgroups, each with thread_execution_width threads.
kernel void batch_rms_norm_f16w(
    device const float* x       [[buffer(0)]],  // [num_tokens, hidden] f32
    device const half*  w       [[buffer(1)]],  // [hidden] f16 weight
    device       float* out     [[buffer(2)]],  // [num_tokens, hidden] f32
    constant     uint&  num_tokens [[buffer(3)]],
    constant     uint&  hidden     [[buffer(4)]],
    constant     float&  eps       [[buffer(5)]],
    constant     uint&  w_add_one  [[buffer(6)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint gid   [[threadgroup_position_in_grid]],  // token index
    uint tgsize[[threads_per_threadgroup]]
) {
    if (gid >= num_tokens) return;

    device const float* x_token = x + gid * hidden;
    device       float* out_token = out + gid * hidden;

    // Phase 1: each thread computes partial sum of squares
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tgsize) {
        float v = x_token[i];
        partial += v * v;
    }

    threadgroup float shared[256];
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree reduction
    for (uint s = tgsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: compute inv_rms
    if (tid == 0) {
        float mean_sq = shared[0] / float(hidden);
        shared[0] = rsqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared[0];

    // Phase 4: apply normalization with weight
    for (uint i = tid; i < hidden; i += tgsize) {
        float wi = float(w[i]);
        if (w_add_one) wi += 1.0f;
        out_token[i] = x_token[i] * inv_rms * wi;
    }
}
"#;

const COPY_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant     uint&  src_offset [[buffer(2)]],
    constant     uint&  dst_offset [[buffer(3)]],
    constant     uint&  n           [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        dst[dst_offset + gid] = src[src_offset + gid];
    }
}
"#;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalOps {
    pub device: Device,
    pub queue: CommandQueue,
    pub _lib: Library,
    pub pso_rms_norm: ComputePipelineState,
    pub pso_rope_adj: ComputePipelineState,
    pub pso_rope_half: ComputePipelineState,
    pub pso_mv_f16: ComputePipelineState,
    pub pso_mv_i8: ComputePipelineState,
    pub pso_mv_i4: ComputePipelineState,
    pub pso_mv_qkv_f16: ComputePipelineState,
    pub pso_mv_qkv_i8: ComputePipelineState,
    pub pso_mv2_f16: ComputePipelineState,
    pub pso_mv2_i8: ComputePipelineState,
    pub pso_add_f32: ComputePipelineState,
    pub pso_mul_f32: ComputePipelineState,
    pub pso_silu_mul_f32: ComputePipelineState,
    pub pso_gelu_mul_f32: ComputePipelineState,
    pub pso_mv_q1: ComputePipelineState,
    pub pso_lfm_conv: ComputePipelineState,
    pub pso_mv_f32: ComputePipelineState,
    pub pso_attention_gqa: ComputePipelineState,
    pub pso_attention_gqa_fast: ComputePipelineState,
    pub pso_scatter_f32: ComputePipelineState,
    pub pso_copy_f32: ComputePipelineState,
    pso_batch_mv_f16: ComputePipelineState,
    pso_batch_rope_half: ComputePipelineState,
    pso_batch_rope_adj: ComputePipelineState,
    pso_batch_write_kv: ComputePipelineState,
    pso_batch_rms_norm: ComputePipelineState,

    x_buf: Mutex<Option<Buffer>>,
    out_buf: Mutex<Option<Buffer>>,
    tensor_cache: Mutex<HashMap<String, Buffer>>,
    /// Named scratch buffers for batched execution (kept on GPU between ops)
    named_bufs: Mutex<HashMap<String, Buffer>>,

    /// Per-layer persistent KV cache buffers (k, v) — max 128 layers.
    kv_cache_bufs: Mutex<Vec<(Option<Buffer>, Option<Buffer>)>>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Clone for MetalOps {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            _lib: self._lib.clone(),
            pso_rms_norm: self.pso_rms_norm.clone(),
            pso_rope_adj: self.pso_rope_adj.clone(),
            pso_rope_half: self.pso_rope_half.clone(),
            pso_mv_f16: self.pso_mv_f16.clone(),
            pso_mv_i8: self.pso_mv_i8.clone(),
            pso_mv_i4: self.pso_mv_i4.clone(),
            pso_mv_qkv_f16: self.pso_mv_qkv_f16.clone(),
            pso_mv_qkv_i8: self.pso_mv_qkv_i8.clone(),
            pso_mv2_f16: self.pso_mv2_f16.clone(),
            pso_mv2_i8: self.pso_mv2_i8.clone(),
            pso_add_f32: self.pso_add_f32.clone(),
            pso_mul_f32: self.pso_mul_f32.clone(),
            pso_silu_mul_f32: self.pso_silu_mul_f32.clone(),
            pso_gelu_mul_f32: self.pso_gelu_mul_f32.clone(),
            pso_mv_q1: self.pso_mv_q1.clone(),
            pso_lfm_conv: self.pso_lfm_conv.clone(),
            pso_mv_f32: self.pso_mv_f32.clone(),
            pso_attention_gqa: self.pso_attention_gqa.clone(),
            pso_attention_gqa_fast: self.pso_attention_gqa_fast.clone(),
            pso_scatter_f32: self.pso_scatter_f32.clone(),
            pso_copy_f32: self.pso_copy_f32.clone(),
            pso_batch_mv_f16: self.pso_batch_mv_f16.clone(),
            pso_batch_rope_half: self.pso_batch_rope_half.clone(),
            pso_batch_rope_adj: self.pso_batch_rope_adj.clone(),
            pso_batch_write_kv: self.pso_batch_write_kv.clone(),
            pso_batch_rms_norm: self.pso_batch_rms_norm.clone(),
            x_buf: Mutex::new(None),
            out_buf: Mutex::new(None),
            tensor_cache: Mutex::new(HashMap::new()),
            named_bufs: Mutex::new(HashMap::new()),
            kv_cache_bufs: Mutex::new(Vec::new()),
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[derive(Clone)]
pub struct MetalOps;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        anyhow::bail!("MetalOps only supported on macOS/iOS")
    }
    pub fn ensure_layer_kv_cache(&self, _: usize, _: usize, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn write_kv_token(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn get_kv_cache_buffers(&self, _: usize) -> Option<(MetalBuffer, MetalBuffer)> { None }
    pub fn ensure_named_buf(&self, _: &str, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn write_named_buf(&self, _: &str, _: &[f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn read_named_buf(&self, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn get_named_buf(&self, _: &str) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn ensure_tensor_cached(&self, _: &str, _: &[u16]) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn ensure_tensor_cached_f32(&self, _: &str, _: &[f32]) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn ensure_tensor_cached_i8(&self, _: &str, _: &[i8]) -> anyhow::Result<MetalBuffer> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn get_cached_tensor(&self, _: &str) -> Option<MetalBuffer> { None }
    pub fn ensure_tensor_cached_i4(&self, _: &str, _: &[u8], _: &str, _: &[u16]) -> anyhow::Result<(MetalBuffer, MetalBuffer)> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn run_batch<F>(&self, _: F) -> anyhow::Result<()> where F: FnOnce(&()) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn lfm_conv(&self, _: &mut [f32], _: &[f32], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn rms_norm_f16w(&self, _: &[f32], _: &[u16], _: f32, _: bool, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn encode_rms_norm_f16w(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: f32, _: bool) {}
    pub fn rope_half_f32(&self, _: &mut [f32], _: usize, _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn encode_rope_half_f32(&self, _enc: &(), _: &MetalBuffer, _: usize, _: usize, _: usize, _: usize, _: f32) {}
    pub fn encode_rope_half_f32_at(&self, _enc: &(), _: &MetalBuffer, _: u64, _: usize, _: usize, _: usize, _: usize, _: f32) {}
    pub fn rope_adj_f32(&self, _: &mut [f32], _: usize, _: usize, _: usize, _: f32) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn encode_rope_adj_f32(&self, _enc: &(), _: &MetalBuffer, _: usize, _: usize, _: usize, _: f32) {}
    pub fn encode_rope_adj_f32_at(&self, _enc: &(), _: &MetalBuffer, _: u64, _: usize, _: usize, _: usize, _: f32) {}
    pub fn encode_mv_f16_bias(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: Option<&MetalBuffer>, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_mv_f16_bias_at(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: u64, _: Option<&MetalBuffer>, _: u64, _: &MetalBuffer, _: u64, _: usize, _: usize) {}
    pub fn encode_mv_i8_bias(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: Option<&MetalBuffer>, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_mv_i8_bias_at(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: u64, _: Option<&MetalBuffer>, _: u64, _: &MetalBuffer, _: u64, _: usize, _: usize) {}
    pub fn encode_qkv_f16_bias(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: Option<&MetalBuffer>, _: Option<&MetalBuffer>, _: Option<&MetalBuffer>, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn encode_qkv_f16_bias_at(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: u64, _: Option<&MetalBuffer>, _: u64, _: Option<&MetalBuffer>, _: u64, _: Option<&MetalBuffer>, _: u64, _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: usize, _: usize, _: usize) {}
    pub fn encode_qkv_i8_bias(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: Option<&MetalBuffer>, _: Option<&MetalBuffer>, _: Option<&MetalBuffer>, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn encode_qkv_i8_bias_at(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: u64, _: Option<&MetalBuffer>, _: u64, _: Option<&MetalBuffer>, _: u64, _: Option<&MetalBuffer>, _: u64, _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: usize, _: usize, _: usize) {}
    pub fn encode_add_f32_inplace(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize) {}
    pub fn encode_add_f32_inplace_at(&self, _enc: &(), _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: usize) {}
    pub fn encode_mul_f32_inplace(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize) {}
    pub fn encode_scatter_f32(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_copy_f32(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn encode_silu_mul_f32_inplace(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize) {}
    pub fn encode_gelu_tanh_mul_f32_inplace(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: usize) {}
    pub fn encode_rms_norm_f16w_at(&self, _enc: &(), _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: &MetalBuffer, _: u64, _: usize, _: f32, _: bool) {}
    pub fn logits_qkv_f16(&self, _: &[f32], _: &[u16], _: &[u16], _: &[u16], _: usize, _: usize, _: usize, _: usize, _: &str, _: &str, _: &str, _: &mut [f32], _: &mut [f32], _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn logits_f16(&self, _: &[f32], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn logits_i8(&self, _: &[f32], _: &[i8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn logits_q1(&self, _: &[f32], _: &[u8], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn encode_mv_f16(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_batch_mv_f16(&self, _: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn encode_batch_rope_half_f32(&self, _: &(), _: &MetalBuffer, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32) {}
    pub fn encode_batch_rope_adj_f32(&self, _: &(), _: &MetalBuffer, _: usize, _: usize, _: usize, _: usize, _: f32) {}
    pub fn encode_batch_write_kv_f32(&self, _: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: u64, _: usize, _: usize, _: usize) {}
    pub fn encode_batch_rms_norm_f16w(&self, _: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: f32, _: bool) {}
    pub fn encode_mv_i8(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_qkv_f16(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn logits_f32(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn encode_mv_f32(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_attention_gqa_f32(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize, _: usize, _: f32, _: f32, _: usize) {}
    pub fn encode_lfm_conv(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_qkv_i8(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn encode_mv_q1(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize) {}
    pub fn encode_mv_i4(&self, _enc: &(), _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize) {}
    pub fn logits_i4(&self, _: &[f32], _: &[u8], _: &[u16], _: usize, _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn attention_and_proj_i4(&self, _: &[f32], _: &[f32], _: &[f32], _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: f32, _: &[u8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn project_qkv_norm_rope_i4(&self, _: &[f32], _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: usize, _: Option<&[u16]>, _: Option<&[u16]>, _: f32, _: usize, _: usize, _: usize, _: usize, _: usize, _: &mut [f32], _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn attention_and_proj_i4_preloaded_q(&self, _: &str, _: &[f32], _: &[f32], _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: f32, _: &[u8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn attention_and_proj_i4_persistent_kv(&self, _: &[f32], _: &MetalBuffer, _: &MetalBuffer, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: f32, _: &[u8], _: &[u16], _: usize, _: usize, _: &str, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn batch_mv_i4(&self, _: &[f32], _: &[(&[u8], &[u16], usize, usize, usize, &str)], _: &mut [&mut [f32]]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn fused_mlp_i4(&self, _: &[f32], _: &[f32], _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: Option<&[u16]>, _: Option<&str>, _: f32, _: bool, _: usize, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
    pub fn fused_post_attn_residual_mlp_i4(&self, _: &[f32], _: &[f32], _: &[u16], _: f32, _: bool, _: &[u16], _: f32, _: bool, _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: &[u8], _: &[u16], _: usize, _: &str, _: Option<&[u16]>, _: Option<&str>, _: f32, _: bool, _: usize, _: &mut [f32]) -> anyhow::Result<()> {
        anyhow::bail!("Metal not available on this platform")
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalOps {
    pub fn create() -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device"))?;
        let queue = device.new_command_queue();
        let lib = {
            let mut guard = ELEM_OPS_LIB_CACHE.lock().unwrap();
            if guard.is_none() {
                *guard = Some(
                    load_or_compile_metallib(&device, ELEM_OPS_SHADER, false, "cellm_kernels")
                        .map_err(|e| anyhow::anyhow!("Kernel library failed: {e:?}"))?
                );
            }
            guard.as_ref().unwrap().clone()
        };

        let pso_rms_norm = build_pso_ops(&device, &lib, "rms_norm_f16w")?;
        let pso_rope_adj = build_pso_ops(&device, &lib, "rope_adj_f32")?;
        let pso_rope_half = build_pso_ops(&device, &lib, "rope_half_f32")?;
        let pso_mv_f16 = build_pso_ops(&device, &lib, "mv_f16")?;
        let pso_mv_i8 = build_pso_ops(&device, &lib, "mv_i8")?;
        let pso_mv_i4 = build_pso_ops(&device, &lib, "mv_i4")?;
        let pso_mv_qkv_f16 = build_pso_ops(&device, &lib, "mv_qkv_f16")?;
        let pso_mv_qkv_i8 = build_pso_ops(&device, &lib, "mv_qkv_i8")?;
        let pso_mv2_f16 = build_pso_ops(&device, &lib, "mv2_f16")?;
        let pso_mv2_i8 = build_pso_ops(&device, &lib, "mv2_i8")?;
        let pso_add_f32 = build_pso_ops(&device, &lib, "add_f32_inplace")?;
        let pso_mul_f32 = build_pso_ops(&device, &lib, "mul_f32_inplace")?;
        let pso_silu_mul_f32 = build_pso_ops(&device, &lib, "silu_mul_f32_inplace")?;
        let pso_gelu_mul_f32 = build_pso_ops(&device, &lib, "gelu_tanh_mul_f32_inplace")?;
        let pso_mv_q1 = build_pso_ops(&device, &lib, "mv_q1_0_g128")?;
        let pso_lfm_conv = build_pso_ops(&device, &lib, "lfm_conv")?;
        let pso_mv_f32 = build_pso_ops(&device, &lib, "mv_f32")?;
        let pso_attention_gqa = build_pso_ops(&device, &lib, "attention_gqa_f32")?;
        let pso_attention_gqa_fast = build_pso_ops(&device, &lib, "attention_gqa_f32_fast")?;
        let pso_scatter_f32 = build_pso_ops(&device, &lib, "scatter_f32")?;
        let pso_batch_mv_f16 = build_pso_ops(&device, &lib, "batch_mv_f16")?;

        let pso_batch_rope_half = build_pso_ops(&device, &lib, "batch_rope_half_f32")?;
        let pso_batch_rope_adj = build_pso_ops(&device, &lib, "batch_rope_adj_f32")?;
        let pso_batch_write_kv = build_pso_ops(&device, &lib, "batch_write_kv_f32")?;
        let pso_batch_rms_norm = build_pso_ops(&device, &lib, "batch_rms_norm_f16w")?;

        // Compile the copy kernel from its own shader string
        let copy_lib = load_or_compile_metallib(&device, COPY_SHADER, false, "cellm_copy")
            .map_err(|e| anyhow::anyhow!("Copy kernel compile failed: {e:?}"))?;
        let pso_copy_f32 = build_pso_ops(&device, &copy_lib, "copy_f32")?;

        Ok(Self {
            device, queue, _lib: lib.clone(),
            pso_rms_norm, pso_rope_adj, pso_rope_half, pso_mv_f16, pso_mv_i8, pso_mv_i4, pso_mv_qkv_f16, pso_mv_qkv_i8, pso_mv2_f16, pso_mv2_i8,
            pso_add_f32, pso_mul_f32, pso_silu_mul_f32, pso_gelu_mul_f32, pso_mv_q1, pso_lfm_conv, pso_mv_f32, pso_attention_gqa, pso_attention_gqa_fast, pso_scatter_f32, pso_copy_f32, pso_batch_mv_f16,
            pso_batch_rope_half, pso_batch_rope_adj, pso_batch_write_kv, pso_batch_rms_norm,
            x_buf: Mutex::new(None), out_buf: Mutex::new(None),
            tensor_cache: Mutex::new(HashMap::new()),
            named_bufs: Mutex::new(HashMap::new()),
            kv_cache_bufs: Mutex::new(Vec::with_capacity(128)),
        })
    }

    // Named GPU buffer management
    // These let the LFM runner keep data on GPU between operations,
    // eliminating per-op CPU↔GPU copies.

    /// Ensure per-layer persistent KV cache buffers exist.
    /// `kv_dim` is floats per token, `max_tokens` is max sequence length.
    pub fn ensure_layer_kv_cache(&self, layer: usize, kv_dim: usize, max_tokens: usize) -> anyhow::Result<()> {
        let mut caches = self.kv_cache_bufs.lock().unwrap();
        if caches.len() <= layer {
            caches.resize_with(layer + 1, || (None, None));
        }
        let bytes = (kv_dim * max_tokens * 4) as u64;
        if caches[layer].0.is_none() || caches[layer].0.as_ref().unwrap().length() < bytes {
            caches[layer].0 = Some(self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared));
        }
        if caches[layer].1.is_none() || caches[layer].1.as_ref().unwrap().length() < bytes {
            caches[layer].1 = Some(self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared));
        }
        Ok(())
    }

    /// Write a single token's K and V into the persistent GPU cache at position `pos`.
    /// Buffers store f32 (4 bytes/elem) to match the attention kernel's expected format.
    pub fn write_kv_token(&self, layer: usize, pos: usize, kv_dim: usize, k_token: &[f32], v_token: &[f32]) -> anyhow::Result<()> {
        let caches = self.kv_cache_bufs.lock().unwrap();
        let entry = caches.get(layer).ok_or_else(|| anyhow::anyhow!("KV cache not initialized for layer {}", layer))?;
        let kbuf: &Buffer = entry.0.as_ref().ok_or_else(|| anyhow::anyhow!("KV cache K not initialized for layer {}", layer))?;
        let vbuf: &Buffer = entry.1.as_ref().ok_or_else(|| anyhow::anyhow!("KV cache V not initialized for layer {}", layer))?;
        assert_eq!(k_token.len(), kv_dim);
        assert_eq!(v_token.len(), kv_dim);
        unsafe {
            let k_dst = (kbuf.contents() as *mut f32).add(pos * kv_dim);
            std::ptr::copy_nonoverlapping(k_token.as_ptr(), k_dst, kv_dim);
            let v_dst = (vbuf.contents() as *mut f32).add(pos * kv_dim);
            std::ptr::copy_nonoverlapping(v_token.as_ptr(), v_dst, kv_dim);
        }
        // On shared/managed memory, no explicit commit needed for CPU writes to be visible to GPU
        Ok(())
    }

    pub fn get_kv_cache_buffers(&self, layer: usize) -> Option<(Buffer, Buffer)> {
        let caches = self.kv_cache_bufs.lock().unwrap();
        let (k_opt, v_opt) = caches.get(layer)?.clone();
        Some((k_opt?, v_opt?))
    }

    /// Ensure a named GPU buffer exists with at least `elems` f32 capacity.
    pub fn ensure_named_buf(&self, name: &str, elems: usize) -> anyhow::Result<()> {
        let need = (elems * 4) as u64;
        let mut bufs = self.named_bufs.lock().unwrap();
        if bufs.get(name).map(|b| b.length() >= need).unwrap_or(false) {
            return Ok(());
        }
        let b = self.device.new_buffer(need, MTLResourceOptions::StorageModeShared);
        if b.contents().is_null() { anyhow::bail!("named buf alloc failed: {name}"); }
        bufs.insert(name.to_string(), b);
        Ok(())
    }

    /// Write f32 data from CPU into a named GPU buffer.
    pub fn write_named_buf(&self, name: &str, src: &[f32]) -> anyhow::Result<()> {
        let bufs = self.named_bufs.lock().unwrap();
        let buf = bufs.get(name).ok_or_else(|| anyhow::anyhow!("no named buf: {name}"))?;
        write_f32(buf, src)
    }

    /// Read f32 data from a named GPU buffer back to CPU.
    pub fn read_named_buf(&self, name: &str, dst: &mut [f32]) -> anyhow::Result<()> {
        let bufs = self.named_bufs.lock().unwrap();
        let buf = bufs.get(name).ok_or_else(|| anyhow::anyhow!("no named buf: {name}"))?;
        read_f32(buf, dst)
    }

    /// Get a clone of a named buffer reference for use in encoding.
    pub fn get_named_buf(&self, name: &str) -> anyhow::Result<Buffer> {
        let bufs = self.named_bufs.lock().unwrap();
        bufs.get(name).cloned().ok_or_else(|| anyhow::anyhow!("no named buf: {name}"))
    }

    /// Cache a tensor (f16 weights) if not already cached, return the buffer.
    pub fn ensure_tensor_cached(&self, key: &str, data_f16: &[u16]) -> anyhow::Result<Buffer> {
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(key) {
            cache.insert(key.to_string(), upload_u16(&self.device, data_f16)?);
        }
        Ok(cache.get(key).unwrap().clone())
    }

    /// Cache an f32 tensor if not already cached, return the buffer.
    pub fn ensure_tensor_cached_f32(&self, key: &str, data: &[f32]) -> anyhow::Result<Buffer> {
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(key) {
            cache.insert(key.to_string(), upload_f32(&self.device, data)?);
        }
        Ok(cache.get(key).unwrap().clone())
    }

    /// Cache an i8 weight buffer if not already cached, return the buffer.
    pub fn ensure_tensor_cached_i8(&self, key: &str, data: &[i8]) -> anyhow::Result<Buffer> {
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(key) {
            cache.insert(key.to_string(), upload_i8(&self.device, data)?);
        }
        Ok(cache.get(key).unwrap().clone())
    }

    /// Retrieve a cloned buffer reference from the tensor cache if present.
    pub fn get_cached_tensor(&self, key: &str) -> Option<Buffer> {
        self.tensor_cache.lock().unwrap().get(key).cloned()
    }

    /// Cache an i4 weight buffer (u8 packed nibbles) and its f16 scales if not already cached.
    pub fn ensure_tensor_cached_i4(&self, w_key: &str, w_u8: &[u8], s_key: &str, s_f16: &[u16]) -> anyhow::Result<(Buffer, Buffer)> {
        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(w_key) {
                cache.insert(w_key.to_string(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(w_u8.as_ptr() as *const i8, w_u8.len())
                })?);
            }
            if !cache.contains_key(s_key) {
                cache.insert(s_key.to_string(), upload_u16(&self.device, s_f16)?);
            }
        }
        let cache = self.tensor_cache.lock().unwrap();
        let w = cache.get(w_key).unwrap().clone();
        let s = cache.get(s_key).unwrap().clone();
        Ok((w, s))
    }

    /// Run a batch of pre-encoded operations in a single command buffer.
    /// `encode_fn` receives the command encoder and encodes all operations.
    /// This is the key to eliminating per-op GPU sync overhead.
    pub fn run_batch<F>(&self, encode_fn: F) -> anyhow::Result<()>
    where
        F: FnOnce(&metal::ComputeCommandEncoderRef) -> anyhow::Result<()>,
    {
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_fn(enc)?;
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            Ok(())
        })
    }

    pub fn lfm_conv(&self, state: &mut [f32], input: &[f32], kernel_f16: &[u16], ks: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let k_key = format!("lfm.k.{}", cache_key);
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(&k_key) {
            cache.insert(k_key.clone(), upload_u16(&self.device, kernel_f16)?);
        }
        let kb = cache.get(&k_key).unwrap().clone();

        let s_key = format!("lfm.state.{}", cache_key);
        if !cache.contains_key(&s_key) {
            cache.insert(s_key.clone(), upload_f32(&self.device, state)?);
        }
        let sb = cache.get(&s_key).unwrap().clone();
        drop(cache);

        // ALWAYS write the current CPU state to the GPU buffer before kernel
        write_f32(&sb, state)?;

        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), hidden)?;
        let xb_lock = self.x_buf.lock().unwrap();
        let ob_lock = self.out_buf.lock().unwrap();
        let xb = xb_lock.as_ref().unwrap();
        let ob = ob_lock.as_ref().unwrap();

        write_f32(xb, input)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_lfm_conv(enc, &sb, xb, &kb, ob, ks, hidden);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Copy state back to CPU
        let ptr = sb.contents() as *const f32;
        unsafe { std::ptr::copy_nonoverlapping(ptr, state.as_mut_ptr(), state.len()); }

        read_f32(ob, out)
    }

    pub fn rms_norm_f16w(&self, x: &[f32], w_f16: &[u16], eps: f32, w_add_one: bool, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let n = x.len();
        let w_key = format!("rmsnorm.w.{cache_key}");
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(&w_key) {
            cache.insert(w_key.clone(), upload_u16(&self.device, w_f16)?);
        }
        let wb = cache.get(&w_key).unwrap().clone();
        drop(cache);
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), n)?;
        let xb_lock = self.x_buf.lock().unwrap(); let xb = xb_lock.as_ref().unwrap();
        let ob_lock = self.out_buf.lock().unwrap(); let ob = ob_lock.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rms_norm_f16w(enc, xb, &wb, ob, n, eps, w_add_one);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn encode_rms_norm_f16w(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, w: &Buffer, out: &Buffer, n: usize, eps: f32, w_add_one: bool) {
        let n32 = n as u32; let add = w_add_one as u32;
        enc.set_compute_pipeline_state(&self.pso_rms_norm);
        enc.set_buffer(10, Some(x), 0); enc.set_buffer(1, Some(w), 0); enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.set_bytes(4, 4, (&eps as *const f32).cast());
        enc.set_bytes(5, 4, (&add as *const u32).cast());
        let tgsize = 256;
        enc.set_threadgroup_memory_length(0, (tgsize * 4) as u64);
        enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: tgsize, height: 1, depth: 1 });
    }

    pub fn rope_half_f32(&self, x: &mut [f32], n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rope_half_f32(enc, xb, n_heads, head_dim, rotary_dim, pos, theta);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }

    pub fn encode_rope_half_f32(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let rd = rotary_dim as u32; let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_half);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&rd as *const u32).cast());
        enc.set_bytes(4, 4, (&p as *const u32).cast());
        enc.set_bytes(5, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (rotary_dim / 2)) as u64;
        let w = self.pso_rope_half.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_rope_half_f32_at(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, x_offset: u64, n_heads: usize, head_dim: usize, rotary_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let rd = rotary_dim as u32; let p = pos as u32;
        let theta = theta as f32;
        enc.set_compute_pipeline_state(&self.pso_rope_half);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&rd as *const u32).cast());
        enc.set_bytes(4, 4, (&p as *const u32).cast());
        enc.set_bytes(5, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (rotary_dim / 2)) as u64;
        let w = self.pso_rope_half.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn rope_adj_f32(&self, x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) -> anyhow::Result<()> {
        let n = x.len(); ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), n)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_rope_adj_f32(enc, xb, n_heads, head_dim, pos, theta);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(xb, x)
    }

    pub fn encode_rope_adj_f32(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_adj);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&p as *const u32).cast());
        enc.set_bytes(4, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (head_dim / 2)) as u64;
        let w = self.pso_rope_adj.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_rope_adj_f32_at(&self, enc: &metal::ComputeCommandEncoderRef, x: &Buffer, x_offset: u64, n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
        let nh = n_heads as u32; let hd = head_dim as u32; let p = pos as u32;
        enc.set_compute_pipeline_state(&self.pso_rope_adj);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_bytes(1, 4, (&nh as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&p as *const u32).cast());
        enc.set_bytes(4, 4, (&theta as *const f32).cast());
        let threads = (n_heads * (head_dim / 2)) as u64;
        let w = self.pso_rope_adj.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_mv_f16_bias(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, b: Option<&metal::Buffer>, out: &metal::Buffer, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_f16);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(x), 0);
        if let Some(bb) = b { enc.set_buffer(2, Some(bb), 0); }
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&r as *const u32).cast());
        enc.set_bytes(5, 4, (&c as *const u32).cast());
        enc.set_bytes(6, 4, (&hb as *const u32).cast());
        let w = self.pso_mv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: w.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_mv_f16_bias_at(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, x_offset: u64, b: Option<&metal::Buffer>, b_offset: u64, out: &metal::Buffer, out_offset: u64, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_f16);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(x), x_offset);
        if let Some(bb) = b { enc.set_buffer(2, Some(bb), b_offset); }
        enc.set_buffer(3, Some(out), out_offset);
        enc.set_bytes(4, 4, (&r as *const u32).cast());
        enc.set_bytes(5, 4, (&c as *const u32).cast());
        enc.set_bytes(6, 4, (&hb as *const u32).cast());
        let w = self.pso_mv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: w.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_mv_i8_bias(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, b: Option<&metal::Buffer>, out: &metal::Buffer, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_i8);
        enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(s), 0); enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(b.unwrap_or(x)), 0);
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, 4, (&r as *const u32).cast());
        enc.set_bytes(6, 4, (&c as *const u32).cast());
        enc.set_bytes(7, 4, (&hb as *const u32).cast());
        let pc = (s.length() >= (rs * 2) as u64) as u32;
        enc.set_bytes(8, 4, (&pc as *const u32).cast());
        let ww = self.pso_mv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: ww.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_mv_i8_bias_at(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, x_offset: u64, b: Option<&metal::Buffer>, b_offset: u64, out: &metal::Buffer, out_offset: u64, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32; let hb = b.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_i8);
        enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(s), 0); enc.set_buffer(2, Some(x), x_offset);
        enc.set_buffer(3, Some(b.unwrap_or(x)), b_offset);
        enc.set_buffer(4, Some(out), out_offset);
        enc.set_bytes(5, 4, (&r as *const u32).cast());
        enc.set_bytes(6, 4, (&c as *const u32).cast());
        enc.set_bytes(7, 4, (&hb as *const u32).cast());
        let pc = (s.length() >= (rs * 2) as u64) as u32;
        enc.set_bytes(8, 4, (&pc as *const u32).cast());
        let ww = self.pso_mv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: ww.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_qkv_f16_bias(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, wk: &metal::Buffer, wv: &metal::Buffer, x: &metal::Buffer, bq: Option<&metal::Buffer>, bk: Option<&metal::Buffer>, bv: Option<&metal::Buffer>, qo: &metal::Buffer, ko: &metal::Buffer, vo: &metal::Buffer, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_f16);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(wk), 0); enc.set_buffer(2, Some(wv), 0); enc.set_buffer(3, Some(x), 0);
        enc.set_buffer(4, Some(bq.unwrap_or(x)), 0);
        enc.set_buffer(5, Some(bk.unwrap_or(x)), 0);
        enc.set_buffer(6, Some(bv.unwrap_or(x)), 0);
        enc.set_buffer(7, Some(qo), 0); enc.set_buffer(8, Some(ko), 0); enc.set_buffer(9, Some(vo), 0);
        enc.set_bytes(10, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(11, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(12, 4, (&c32 as *const u32).cast());
        enc.set_bytes(13, 4, (&hb as *const u32).cast());
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_qkv_f16_bias_at(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, wk: &metal::Buffer, wv: &metal::Buffer, x: &metal::Buffer, x_offset: u64, bq: Option<&metal::Buffer>, bq_offset: u64, bk: Option<&metal::Buffer>, bk_offset: u64, bv: Option<&metal::Buffer>, bv_offset: u64, qo: &metal::Buffer, qo_offset: u64, ko: &metal::Buffer, ko_offset: u64, vo: &metal::Buffer, vo_offset: u64, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_f16);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(wk), 0); enc.set_buffer(2, Some(wv), 0); enc.set_buffer(3, Some(x), x_offset);
        enc.set_buffer(4, Some(bq.unwrap_or(x)), bq_offset);
        enc.set_buffer(5, Some(bk.unwrap_or(x)), bk_offset);
        enc.set_buffer(6, Some(bv.unwrap_or(x)), bv_offset);
        enc.set_buffer(7, Some(qo), qo_offset); enc.set_buffer(8, Some(ko), ko_offset); enc.set_buffer(9, Some(vo), vo_offset);
        enc.set_bytes(10, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(11, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(12, 4, (&c32 as *const u32).cast());
        enc.set_bytes(13, 4, (&hb as *const u32).cast());
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_f16.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_qkv_i8_bias(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, sq: &metal::Buffer, wk: &metal::Buffer, sk: &metal::Buffer, wv: &metal::Buffer, sv: &metal::Buffer, x: &metal::Buffer, bq: Option<&metal::Buffer>, bk: Option<&metal::Buffer>, bv: Option<&metal::Buffer>, qo: &metal::Buffer, ko: &metal::Buffer, vo: &metal::Buffer, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_i8);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(sq), 0);
        enc.set_buffer(2, Some(wk), 0); enc.set_buffer(3, Some(sk), 0);
        enc.set_buffer(4, Some(wv), 0); enc.set_buffer(5, Some(sv), 0);
        enc.set_buffer(6, Some(x), 0);
        enc.set_buffer(7, Some(bq.unwrap_or(x)), 0);
        enc.set_buffer(8, Some(bk.unwrap_or(x)), 0);
        enc.set_buffer(9, Some(bv.unwrap_or(x)), 0);
        enc.set_buffer(10, Some(qo), 0); enc.set_buffer(11, Some(ko), 0); enc.set_buffer(12, Some(vo), 0);
        enc.set_bytes(13, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(14, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(15, 4, (&c32 as *const u32).cast());
        enc.set_bytes(16, 4, (&hb as *const u32).cast());
        let pc_q = (sq.length() >= (rq * 2) as u64) as u32;
        let pc_k = (sk.length() >= (rkv * 2) as u64) as u32;
        let pc_v = (sv.length() >= (rkv * 2) as u64) as u32;
        enc.set_bytes(17, 16, [pc_q, pc_k, pc_v, 0u32].as_ptr() as *const _);
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_qkv_i8_bias_at(&self, enc: &metal::ComputeCommandEncoderRef, wq: &metal::Buffer, sq: &metal::Buffer, wk: &metal::Buffer, sk: &metal::Buffer, wv: &metal::Buffer, sv: &metal::Buffer, x: &metal::Buffer, x_offset: u64, bq: Option<&metal::Buffer>, bq_offset: u64, bk: Option<&metal::Buffer>, bk_offset: u64, bv: Option<&metal::Buffer>, bv_offset: u64, qo: &metal::Buffer, qo_offset: u64, ko: &metal::Buffer, ko_offset: u64, vo: &metal::Buffer, vo_offset: u64, rq: usize, rkv: usize, c: usize) {
        let rq32 = rq as u32; let rkv32 = rkv as u32; let c32 = c as u32; let hb = bq.is_some() as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_qkv_i8);
        enc.set_buffer(0, Some(wq), 0); enc.set_buffer(1, Some(sq), 0);
        enc.set_buffer(2, Some(wk), 0); enc.set_buffer(3, Some(sk), 0);
        enc.set_buffer(4, Some(wv), 0); enc.set_buffer(5, Some(sv), 0);
        enc.set_buffer(6, Some(x), x_offset);
        enc.set_buffer(7, Some(bq.unwrap_or(x)), bq_offset);
        enc.set_buffer(8, Some(bk.unwrap_or(x)), bk_offset);
        enc.set_buffer(9, Some(bv.unwrap_or(x)), bv_offset);
        enc.set_buffer(10, Some(qo), qo_offset); enc.set_buffer(11, Some(ko), ko_offset); enc.set_buffer(12, Some(vo), vo_offset);
        enc.set_bytes(13, 4, (&rq32 as *const u32).cast());
        enc.set_bytes(14, 4, (&rkv32 as *const u32).cast());
        enc.set_bytes(15, 4, (&c32 as *const u32).cast());
        enc.set_bytes(16, 4, (&hb as *const u32).cast());
        let pc_q = (sq.length() >= (rq * 2) as u64) as u32;
        let pc_k = (sk.length() >= (rkv * 2) as u64) as u32;
        let pc_v = (sv.length() >= (rkv * 2) as u64) as u32;
        enc.set_bytes(17, 16, [pc_q, pc_k, pc_v, 0u32].as_ptr() as *const _);
        let threads = (rq + rkv + rkv) as u64;
        let w = self.pso_mv_qkv_i8.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_add_f32_inplace(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, b: &metal::BufferRef, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_add_f32);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_add_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    pub fn encode_add_f32_inplace_at(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, a_offset: u64, b: &metal::BufferRef, b_offset: u64, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_add_f32);
        enc.set_buffer(0, Some(a), a_offset); enc.set_buffer(1, Some(b), b_offset);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_add_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    pub fn encode_mul_f32_inplace(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, b: &metal::BufferRef, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_mul_f32);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_mul_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    pub fn encode_scatter_f32(&self, enc: &metal::ComputeCommandEncoderRef, src: &metal::BufferRef, dst: &metal::BufferRef, offset: usize, n: usize) {
        let off = offset as u32;
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_scatter_f32);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, (&off as *const u32).cast());
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        let threads = n as u64;
        let w = self.pso_scatter_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_copy_f32(&self, enc: &metal::ComputeCommandEncoderRef, src: &metal::BufferRef, dst: &metal::BufferRef, src_offset: usize, dst_offset: usize, n: usize) {
        let src_off = src_offset as u32;
        let dst_off = dst_offset as u32;
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_copy_f32);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, (&src_off as *const u32).cast());
        enc.set_bytes(3, 4, (&dst_off as *const u32).cast());
        enc.set_bytes(4, 4, (&n32 as *const u32).cast());
        let threads = n as u64;
        let w = self.pso_copy_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: w.min(threads), height: 1, depth: 1 });
    }

    pub fn encode_silu_mul_f32_inplace(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::BufferRef, b: &metal::BufferRef, n: usize) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_silu_mul_f32);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_silu_mul_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: n as u64, height: 1, depth: 1 }, MTLSize { width: w.min(n as u64), height: 1, depth: 1 });
    }

    /// GELU(gate)*up in-place — Gemma's activation function.
    pub fn encode_gelu_tanh_mul_f32_inplace(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        gate: &metal::BufferRef,
        up: &metal::BufferRef,
        n: usize,
    ) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.pso_gelu_mul_f32);
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
        let w = self.pso_gelu_mul_f32.thread_execution_width() as u64;
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: w.min(n as u64), height: 1, depth: 1 },
        );
    }

    /// RMSNorm with byte-offset into x and out — used for per-head Q/K norms in
    /// Gemma where one weight vector is applied independently to each head's slice.
    pub fn encode_rms_norm_f16w_at(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &metal::BufferRef,
        x_byte_offset: u64,
        w: &metal::BufferRef,
        w_byte_offset: u64,
        out: &metal::BufferRef,
        out_byte_offset: u64,
        n: usize,
        eps: f32,
        w_add_one: bool,
    ) {
        let n32 = n as u32;
        let add = w_add_one as u32;
        enc.set_compute_pipeline_state(&self.pso_rms_norm);
        enc.set_buffer(10, Some(x), x_byte_offset);
        enc.set_buffer(1, Some(w), w_byte_offset);
        enc.set_buffer(2, Some(out), out_byte_offset);
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.set_bytes(4, 4, (&eps as *const f32).cast());
        enc.set_bytes(5, 4, (&add as *const u32).cast());
        let tgsize: u64 = 256;
        enc.set_threadgroup_memory_length(0, tgsize * 4);
        enc.dispatch_thread_groups(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: tgsize, height: 1, depth: 1 },
        );
    }

    pub fn logits_qkv_f16(&self, x: &[f32], q_w: &[u16], k_w: &[u16], v_w: &[u16], q_dim: usize, k_dim: usize, v_dim: usize, hidden: usize, q_key: &str, k_key: &str, v_key: &str, q_out: &mut [f32], k_out: &mut [f32], v_out: &mut [f32]) -> anyhow::Result<()> {
        self.logits_f16(x, q_w, q_dim, hidden, q_key, q_out)?;
        self.logits_f16(x, k_w, k_dim, hidden, k_key, k_out)?;
        self.logits_f16(x, v_w, v_dim, hidden, v_key, v_out)?;
        Ok(())
    }

    pub fn logits_f16(&self, x: &[f32], embed_f16: &[u16], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key) {
            self.tensor_cache.lock().unwrap().insert(cache_key.to_string(), upload_u16(&self.device, embed_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(cache_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_f16_bias(enc, eb, xb, None, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn logits_i8(&self, x: &[f32], embed_i8: &[i8], scales_f16: &[u16], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let w_key = format!("logits.w.{cache_key}");
        let s_key = format!("logits.s.{cache_key}");
        if !self.tensor_cache.lock().unwrap().contains_key(&w_key) {
            self.tensor_cache.lock().unwrap().insert(w_key.clone(), upload_i8(&self.device, embed_i8)?);
            self.tensor_cache.lock().unwrap().insert(s_key.clone(), upload_u16(&self.device, scales_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(&w_key).unwrap();
        let sb = cache.get(&s_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_i8_bias(enc, eb, sb, xb, None, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn logits_q1(&self, x: &[f32], embed_q1: &[u8], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key) {
            self.tensor_cache.lock().unwrap().insert(cache_key.to_string(), upload_i8(&self.device, unsafe { std::slice::from_raw_parts(embed_q1.as_ptr() as *const i8, embed_q1.len()) })?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(cache_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_q1(enc, eb, xb, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    // Compat stubs for old callers
    pub fn encode_mv_f16(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize) {
        self.encode_mv_f16_bias(enc, a, x, None, out, rs, cs);
    }
    pub fn encode_mv_i8(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize) {
        self.encode_mv_i8_bias(enc, w, s, x, None, out, rs, cs);
    }
    pub fn encode_qkv_f16(&self, enc: &metal::ComputeCommandEncoderRef, w_q: &metal::Buffer, w_k: &metal::Buffer, w_v: &metal::Buffer, x_buf: &metal::Buffer, q_out: &metal::Buffer, k_out: &metal::Buffer, v_out: &metal::Buffer, rows_q: usize, rows_kv: usize, cols: usize) {
        self.encode_qkv_f16_bias(enc, w_q, w_k, w_v, x_buf, None, None, None, q_out, k_out, v_out, rows_q, rows_kv, cols);
    }

    pub fn encode_batch_mv_f16(&self, enc: &metal::ComputeCommandEncoderRef,
        weight: &metal::Buffer, x_all: &metal::Buffer, out_all: &metal::Buffer,
        num_tokens: usize, out_dim: usize, in_dim: usize)
    {
        let nt = num_tokens as u32;
        let od = out_dim as u32;
        let id = in_dim as u32;
        enc.set_compute_pipeline_state(&self.pso_batch_mv_f16);
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(x_all), 0);
        enc.set_buffer(2, Some(out_all), 0);
        enc.set_bytes(3, 4, (&nt as *const u32).cast());
        enc.set_bytes(4, 4, (&od as *const u32).cast());
        enc.set_bytes(5, 4, (&id as *const u32).cast());
        let grid = MTLSize { width: out_dim as u64, height: num_tokens as u64, depth: 1 };
        let w = self.pso_batch_mv_f16.thread_execution_width() as u64;
        let tg = MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_batch_rope_half_f32(&self, enc: &metal::ComputeCommandEncoderRef,
        x: &metal::Buffer, num_tokens: usize, n_heads: usize, head_dim: usize,
        rotary_dim: usize, start_pos: usize, theta: f32)
    {
        let nt = num_tokens as u32;
        let nh = n_heads as u32;
        let hd = head_dim as u32;
        let rd = rotary_dim as u32;
        let sp = start_pos as u32;
        enc.set_compute_pipeline_state(&self.pso_batch_rope_half);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nt as *const u32).cast());
        enc.set_bytes(2, 4, (&nh as *const u32).cast());
        enc.set_bytes(3, 4, (&hd as *const u32).cast());
        enc.set_bytes(4, 4, (&rd as *const u32).cast());
        enc.set_bytes(5, 4, (&sp as *const u32).cast());
        enc.set_bytes(6, 4, (&theta as *const f32).cast());
        let num_pairs = n_heads * (rotary_dim / 2);
        let grid = MTLSize { width: num_pairs as u64, height: num_tokens as u64, depth: 1 };
        let w = self.pso_batch_rope_half.thread_execution_width() as u64;
        let tg = MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_batch_rope_adj_f32(&self, enc: &metal::ComputeCommandEncoderRef,
        x: &metal::Buffer, num_tokens: usize, n_heads: usize, head_dim: usize,
        start_pos: usize, theta: f32)
    {
        let nt = num_tokens as u32;
        let nh = n_heads as u32;
        let hd = head_dim as u32;
        let sp = start_pos as u32;
        enc.set_compute_pipeline_state(&self.pso_batch_rope_adj);
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, (&nt as *const u32).cast());
        enc.set_bytes(2, 4, (&nh as *const u32).cast());
        enc.set_bytes(3, 4, (&hd as *const u32).cast());
        enc.set_bytes(4, 4, (&sp as *const u32).cast());
        enc.set_bytes(5, 4, (&theta as *const f32).cast());
        let num_pairs = n_heads * (head_dim / 2);
        let grid = MTLSize { width: num_pairs as u64, height: num_tokens as u64, depth: 1 };
        let w = self.pso_batch_rope_adj.thread_execution_width() as u64;
        let tg = MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    /// Write all tokens' K/V to KV cache using batch kernel.
    /// bases[pos] = base element in KV cache for position pos (layer-relative).
    pub fn encode_batch_write_kv_f32(&self, enc: &metal::ComputeCommandEncoderRef,
        k_cache: &metal::BufferRef, v_cache: &metal::BufferRef,
        k_all: &metal::Buffer, v_all: &metal::Buffer,
        bases: &metal::Buffer, bases_offset: u64,
        num_tokens: usize, kv_dim: usize, start_pos: usize)
    {
        let nt = num_tokens as u32;
        let kd = kv_dim as u32;
        let sp = start_pos as u32;
        enc.set_compute_pipeline_state(&self.pso_batch_write_kv);
        enc.set_buffer(0, Some(k_cache), 0);
        enc.set_buffer(1, Some(v_cache), 0);
        enc.set_buffer(2, Some(k_all), 0);
        enc.set_buffer(3, Some(v_all), 0);
        enc.set_buffer(4, Some(bases), bases_offset);
        enc.set_bytes(5, 4, (&nt as *const u32).cast());
        enc.set_bytes(6, 4, (&kd as *const u32).cast());
        enc.set_bytes(7, 4, (&sp as *const u32).cast());
        let grid = MTLSize { width: kv_dim as u64, height: num_tokens as u64, depth: 1 };
        let w = self.pso_batch_write_kv.thread_execution_width() as u64;
        let tg = MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
    }

    pub fn encode_batch_rms_norm_f16w(&self, enc: &metal::ComputeCommandEncoderRef,
        x_all: &metal::BufferRef, w: &metal::BufferRef, out_all: &metal::BufferRef,
        num_tokens: usize, hidden: usize, eps: f32, w_add_one: bool)
    {
        let nt = num_tokens as u32;
        let hd = hidden as u32;
        let wa = w_add_one as u32;
        enc.set_compute_pipeline_state(&self.pso_batch_rms_norm);
        enc.set_buffer(0, Some(x_all), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out_all), 0);
        enc.set_bytes(3, 4, (&nt as *const u32).cast());
        enc.set_bytes(4, 4, (&hd as *const u32).cast());
        enc.set_bytes(5, 4, (&eps as *const f32).cast());
        enc.set_bytes(6, 4, (&wa as *const u32).cast());
        let w = self.pso_batch_rms_norm.thread_execution_width() as u64;
        let tg = MTLSize { width: w, height: 1, depth: 1 };
        let grid = MTLSize { width: num_tokens as u64, height: 1, depth: 1 };
        enc.dispatch_thread_groups(grid, tg);
    }

    pub fn logits_f32(&self, x: &[f32], weights_f32: &[f32], vocab: usize, hidden: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        if !self.tensor_cache.lock().unwrap().contains_key(cache_key) {
            self.tensor_cache.lock().unwrap().insert(cache_key.to_string(), upload_f32(&self.device, weights_f32)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let eb = cache.get(cache_key).unwrap();
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_f32(enc, eb, xb, ob, vocab, hidden);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    pub fn encode_mv_f32(&self, enc: &metal::ComputeCommandEncoderRef, w: &Buffer, x: &Buffer, out: &Buffer, rows: usize, cols: usize) {
        let r = rows as u32; let c = cols as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_f32);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, (&r as *const u32).cast());
        enc.set_bytes(4, 4, (&c as *const u32).cast());
        let w_size = self.pso_mv_f32.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rows as u64, height: 1, depth: 1 }, MTLSize { width: w_size, height: 1, depth: 1 });
    }

    pub fn encode_attention_gqa_f32(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        scale: f32,
        soft_cap: f32,
        kv_stride: usize,
    ) {
        let nh = n_heads as u32;
        let nkv = n_kv_heads as u32;
        let hd = head_dim as u32;
        let sl = seq_len as u32;
        let sc = scale as f32;
        let cap = soft_cap as f32;
        let ks = kv_stride as u32;
        enc.set_compute_pipeline_state(&self.pso_attention_gqa_fast);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), 0);
        enc.set_buffer(2, Some(v), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&nh as *const u32).cast());
        enc.set_bytes(5, 4, (&nkv as *const u32).cast());
        enc.set_bytes(6, 4, (&hd as *const u32).cast());
        enc.set_bytes(7, 4, (&sl as *const u32).cast());
        enc.set_bytes(8, 4, (&sc as *const f32).cast());
        enc.set_bytes(9, 4, (&cap as *const f32).cast());
        enc.set_bytes(10, 4, (&ks as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize { width: n_heads as u64, height: 1, depth: 1 },
            MTLSize { width: 128, height: 1, depth: 1 },
        );
    }

    pub fn encode_lfm_conv(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        state: &metal::Buffer,
        input: &metal::Buffer,
        kernel: &metal::Buffer,
        out: &metal::Buffer,
        ks: usize,
        hidden: usize,
    ) {
        let ks32 = ks as u32;
        let h32 = hidden as u32;
        enc.set_compute_pipeline_state(&self.pso_lfm_conv);
        enc.set_buffer(0, Some(state), 0);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(kernel), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&ks32 as *const u32).cast());
        enc.set_bytes(5, 4, (&h32 as *const u32).cast());
        let w = (256u64).min(hidden as u64);
        enc.dispatch_threads(MTLSize { width: hidden as u64, height: 1, depth: 1 }, MTLSize { width: w, height: 1, depth: 1 });
    }
    pub fn encode_qkv_i8(&self, enc: &metal::ComputeCommandEncoderRef, w_q: &metal::Buffer, s_q: &metal::Buffer, w_k: &metal::Buffer, s_k: &metal::Buffer, w_v: &metal::Buffer, s_v: &metal::Buffer, x_buf: &metal::Buffer, q_out: &metal::Buffer, k_out: &metal::Buffer, v_out: &metal::Buffer, rows_q: usize, rows_kv: usize, cols: usize) {
        self.encode_qkv_i8_bias(enc, w_q, s_q, w_k, s_k, w_v, s_v, x_buf, None, None, None, q_out, k_out, v_out, rows_q, rows_kv, cols);
    }

    pub fn encode_mv_q1(&self, enc: &metal::ComputeCommandEncoderRef, a: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize) {
        let r = rs as u32; let c = cs as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_q1);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(x), 0); enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, (&r as *const u32).cast());
        enc.set_bytes(4, 4, (&c as *const u32).cast());
        let w = self.pso_mv_q1.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: w.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn encode_mv_i4(&self, enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer, s: &metal::Buffer, x: &metal::Buffer, out: &metal::Buffer, rs: usize, cs: usize, gs: usize) {
        let r = rs as u32; let c = cs as u32; let g = gs as u32;
        enc.set_compute_pipeline_state(&self.pso_mv_i4);
        enc.set_buffer(0, Some(w), 0); enc.set_buffer(1, Some(s), 0); enc.set_buffer(2, Some(x), 0); enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, (&r as *const u32).cast());
        enc.set_bytes(5, 4, (&c as *const u32).cast());
        enc.set_bytes(6, 4, (&g as *const u32).cast());
        let w = self.pso_mv_i4.thread_execution_width() as u64;
        enc.dispatch_threads(MTLSize { width: rs as u64, height: 1, depth: 1 }, MTLSize { width: w.min(rs as u64), height: 1, depth: 1 });
    }

    pub fn logits_i4(&self, x: &[f32], w_u8: &[u8], s_f16: &[u16], vocab: usize, hidden: usize, gs: usize, cache_key: &str, out: &mut [f32]) -> anyhow::Result<()> {
        let w_key = format!("logits.w.{}", cache_key);
        let s_key = format!("logits.s.{}", cache_key);
        if !self.tensor_cache.lock().unwrap().contains_key(&w_key) {
            self.tensor_cache.lock().unwrap().insert(w_key.clone(), upload_i8(&self.device, unsafe { std::slice::from_raw_parts(w_u8.as_ptr() as *const i8, w_u8.len()) })?);
        }
        if !self.tensor_cache.lock().unwrap().contains_key(&s_key) {
            self.tensor_cache.lock().unwrap().insert(s_key.clone(), upload_u16(&self.device, s_f16)?);
        }
        let cache = self.tensor_cache.lock().unwrap();
        let wb = cache.get(&w_key).unwrap();
        let sb = cache.get(&s_key).unwrap();

        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), hidden)?;
        ensure_buf_f32(&self.device, &mut *self.out_buf.lock().unwrap(), vocab)?;
        let xb_ref = self.x_buf.lock().unwrap(); let xb = xb_ref.as_ref().unwrap();
        let ob_ref = self.out_buf.lock().unwrap(); let ob = ob_ref.as_ref().unwrap();
        write_f32(xb, x)?;

        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.encode_mv_i4(enc, wb, sb, xb, ob, vocab, hidden, gs);
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        });
        read_f32(ob, out)
    }

    /// Run GQA attention on GPU followed by i4 o_proj matmul, all in one command buffer.
    /// q: [n_heads, head_dim] f32
    /// k/v: [seq_len, kv_stride] f32 (gathered from KV cache, may be padded)
    /// o_proj: i4 weight [hidden, q_dim]
    pub fn attention_and_proj_i4(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        attn_scale: f32,
        soft_cap: f32,
        o_proj_w: &[u8],
        o_proj_s: &[u16],
        o_proj_out_dim: usize,
        o_proj_in_dim: usize,
        o_proj_key: &str,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        let q_dim = n_heads * head_dim;
        assert_eq!(q.len(), q_dim);
        assert!(k.len() >= seq_len * kv_stride, "k.len={} < seq_len*kv_stride={}", k.len(), seq_len * kv_stride);
        assert!(v.len() >= seq_len * kv_stride, "v.len={} < seq_len*kv_stride={}", v.len(), seq_len * kv_stride);
        assert_eq!(out.len(), o_proj_out_dim);

        // 1. Cache o_proj weight.
        let w_key = format!("attn.o_proj.w.{}", o_proj_key);
        let s_key = format!("attn.o_proj.s.{}", o_proj_key);
        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&w_key) {
                cache.insert(w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(o_proj_w.as_ptr() as *const i8, o_proj_w.len())
                })?);
            }
            if !cache.contains_key(&s_key) {
                cache.insert(s_key.clone(), upload_u16(&self.device, o_proj_s)?);
            }
        }

        // 2. Ensure named buffers.
        self.ensure_named_buf("attn_q", q_dim)?;
        self.ensure_named_buf("attn_k", k.len())?;
        self.ensure_named_buf("attn_v", v.len())?;
        self.ensure_named_buf("attn_out", q_dim)?;
        self.ensure_named_buf("attn_proj_out", o_proj_out_dim)?;

        // 3. Upload Q/K/V.
        self.write_named_buf("attn_q", q)?;
        self.write_named_buf("attn_k", k)?;
        self.write_named_buf("attn_v", v)?;

        // 4. Get buffer handles.
        let cache = self.tensor_cache.lock().unwrap();
        let o_proj_wb = cache.get(&w_key).unwrap().clone();
        let o_proj_sb = cache.get(&s_key).unwrap().clone();
        drop(cache);

        let q_buf = self.get_named_buf("attn_q")?;
        let k_buf = self.get_named_buf("attn_k")?;
        let v_buf = self.get_named_buf("attn_v")?;
        let attn_out_buf = self.get_named_buf("attn_out")?;
        let proj_out_buf = self.get_named_buf("attn_proj_out")?;

        // 5. Encode attention + o_proj in one command buffer.
        self.run_batch(|enc| {
            self.encode_attention_gqa_f32(
                enc, &q_buf, &k_buf, &v_buf, &attn_out_buf,
                n_heads, n_kv_heads, head_dim, seq_len,
                attn_scale, soft_cap, kv_stride,
            );
            self.encode_mv_i4(
                enc, &o_proj_wb, &o_proj_sb, &attn_out_buf, &proj_out_buf,
                o_proj_out_dim, o_proj_in_dim, o_proj_in_dim,
            );
            Ok(())
        })?;

        // 6. Read back.
        self.read_named_buf("attn_proj_out", out)?;
        Ok(())
    }

    /// Decode-step helper: QKV i4 projections + per-head Q/K norms + RoPE in one command buffer.
    /// Q stays resident on GPU in named buffer "attn_block_q".
    /// K and V are downloaded to `new_k` / `new_v` so the caller can write them to the CPU KV cache.
    pub fn project_qkv_norm_rope_i4(
        &self,
        x_norm: &[f32],
        q_w: &[u8],
        q_s: &[u16],
        q_dim: usize,
        q_key: &str,
        k_w: &[u8],
        k_s: &[u16],
        k_dim: usize,
        k_key: &str,
        v_w: &[u8],
        v_s: &[u16],
        v_dim: usize,
        v_key: &str,
        hidden: usize,
        q_norm_w: Option<&[u16]>,
        k_norm_w: Option<&[u16]>,
        rope_theta: f32,
        pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_head_dim: usize,
        new_k: &mut [f32],
        new_v: &mut [f32],
    ) -> anyhow::Result<()> {
        assert_eq!(x_norm.len(), hidden);
        assert_eq!(new_k.len(), k_dim);
        assert_eq!(new_v.len(), v_dim);

        let q_w_key = format!("attn.q.w.{}", q_key);
        let q_s_key = format!("attn.q.s.{}", q_key);
        let k_w_key = format!("attn.k.w.{}", k_key);
        let k_s_key = format!("attn.k.s.{}", k_key);
        let v_w_key = format!("attn.v.w.{}", v_key);
        let v_s_key = format!("attn.v.s.{}", v_key);

        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&q_w_key) {
                cache.insert(q_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(q_w.as_ptr() as *const i8, q_w.len())
                })?);
            }
            if !cache.contains_key(&q_s_key) {
                cache.insert(q_s_key.clone(), upload_u16(&self.device, q_s)?);
            }
            if !cache.contains_key(&k_w_key) {
                cache.insert(k_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(k_w.as_ptr() as *const i8, k_w.len())
                })?);
            }
            if !cache.contains_key(&k_s_key) {
                cache.insert(k_s_key.clone(), upload_u16(&self.device, k_s)?);
            }
            if !cache.contains_key(&v_w_key) {
                cache.insert(v_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(v_w.as_ptr() as *const i8, v_w.len())
                })?);
            }
            if !cache.contains_key(&v_s_key) {
                cache.insert(v_s_key.clone(), upload_u16(&self.device, v_s)?);
            }
        }

        let q_norm_key = q_norm_w.map(|_| format!("attn.q_norm.w.{}", q_key));
        let k_norm_key = k_norm_w.map(|_| format!("attn.k_norm.w.{}", k_key));
        if let Some(w) = q_norm_w {
            let key = format!("attn.q_norm.w.{}", q_key);
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&key) {
                cache.insert(key.clone(), upload_u16(&self.device, w)?);
            }
        }
        if let Some(w) = k_norm_w {
            let key = format!("attn.k_norm.w.{}", k_key);
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&key) {
                cache.insert(key.clone(), upload_u16(&self.device, w)?);
            }
        }

        self.ensure_named_buf("attn_block_x", hidden)?;
        self.ensure_named_buf("attn_block_q", q_dim)?;
        self.ensure_named_buf("attn_block_k", k_dim)?;
        self.ensure_named_buf("attn_block_v", v_dim)?;

        self.write_named_buf("attn_block_x", x_norm)?;

        let cache = self.tensor_cache.lock().unwrap();
        let q_wb = cache.get(&q_w_key).unwrap().clone();
        let q_sb = cache.get(&q_s_key).unwrap().clone();
        let k_wb = cache.get(&k_w_key).unwrap().clone();
        let k_sb = cache.get(&k_s_key).unwrap().clone();
        let v_wb = cache.get(&v_w_key).unwrap().clone();
        let v_sb = cache.get(&v_s_key).unwrap().clone();
        let q_norm_wb = q_norm_key.as_ref().and_then(|k| cache.get(k).cloned());
        let k_norm_wb = k_norm_key.as_ref().and_then(|k| cache.get(k).cloned());
        drop(cache);

        let x_buf = self.get_named_buf("attn_block_x")?;
        let q_buf = self.get_named_buf("attn_block_q")?;
        let k_buf = self.get_named_buf("attn_block_k")?;
        let v_buf = self.get_named_buf("attn_block_v")?;

        self.run_batch(|enc| {
            self.encode_mv_i4(enc, &q_wb, &q_sb, &x_buf, &q_buf, q_dim, hidden, hidden);
            self.encode_mv_i4(enc, &k_wb, &k_sb, &x_buf, &k_buf, k_dim, hidden, hidden);
            self.encode_mv_i4(enc, &v_wb, &v_sb, &x_buf, &v_buf, v_dim, hidden, hidden);

            if let Some(ref qw) = q_norm_wb {
                for hidx in 0..n_heads {
                    let off = (hidx * head_dim * 4) as u64;
                    self.encode_rms_norm_f16w_at(enc, &q_buf, off, qw, 0, &q_buf, off, head_dim, 1e-6, false);
                }
            }
            if let Some(ref kw) = k_norm_wb {
                for hidx in 0..n_kv_heads {
                    let off = (hidx * kv_head_dim * 4) as u64;
                    self.encode_rms_norm_f16w_at(enc, &k_buf, off, kw, 0, &k_buf, off, kv_head_dim, 1e-6, false);
                }
            }

            self.encode_rope_half_f32(enc, &q_buf, n_heads, head_dim, head_dim, pos, rope_theta);
            self.encode_rope_half_f32(enc, &k_buf, n_kv_heads, kv_head_dim, kv_head_dim, pos, rope_theta);
            Ok(())
        })?;

        self.read_named_buf("attn_block_k", new_k)?;
        self.read_named_buf("attn_block_v", new_v)?;
        Ok(())
    }

    /// Attention + o_proj where Q is already resident on GPU (e.g. from `project_qkv_norm_rope_i4`).
    /// K/V sequences are uploaded from CPU buffers gathered from the KV cache.
    pub fn attention_and_proj_i4_preloaded_q(
        &self,
        q_buf_name: &str,
        k_seq: &[f32],
        v_seq: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        attn_scale: f32,
        soft_cap: f32,
        o_proj_w: &[u8],
        o_proj_s: &[u16],
        o_proj_out_dim: usize,
        o_proj_in_dim: usize,
        o_proj_key: &str,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        let q_dim = n_heads * head_dim;
        assert_eq!(out.len(), o_proj_out_dim);

        let q_buf = self.get_named_buf(q_buf_name)?;

        self.ensure_named_buf("attn_k_preload", k_seq.len())?;
        self.ensure_named_buf("attn_v_preload", v_seq.len())?;
        self.ensure_named_buf("attn_out_preload", q_dim)?;
        self.ensure_named_buf("attn_proj_out_preload", o_proj_out_dim)?;
        self.write_named_buf("attn_k_preload", k_seq)?;
        self.write_named_buf("attn_v_preload", v_seq)?;

        let k_buf = self.get_named_buf("attn_k_preload")?;
        let v_buf = self.get_named_buf("attn_v_preload")?;
        let attn_out_buf = self.get_named_buf("attn_out_preload")?;
        let proj_out_buf = self.get_named_buf("attn_proj_out_preload")?;

        let w_key = format!("attn.o_proj.w.{}", o_proj_key);
        let s_key = format!("attn.o_proj.s.{}", o_proj_key);
        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&w_key) {
                cache.insert(w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(o_proj_w.as_ptr() as *const i8, o_proj_w.len())
                })?);
            }
            if !cache.contains_key(&s_key) {
                cache.insert(s_key.clone(), upload_u16(&self.device, o_proj_s)?);
            }
        }

        let cache = self.tensor_cache.lock().unwrap();
        let o_proj_wb = cache.get(&w_key).unwrap().clone();
        let o_proj_sb = cache.get(&s_key).unwrap().clone();
        drop(cache);

        self.run_batch(|enc| {
            self.encode_attention_gqa_f32(
                enc, &q_buf, &k_buf, &v_buf, &attn_out_buf,
                n_heads, n_kv_heads, head_dim, seq_len,
                attn_scale, soft_cap, kv_stride,
            );
            self.encode_mv_i4(
                enc, &o_proj_wb, &o_proj_sb, &attn_out_buf, &proj_out_buf,
                o_proj_out_dim, o_proj_in_dim, o_proj_in_dim,
            );
            Ok(())
        })?;

        self.read_named_buf("attn_proj_out_preload", out)?;
        Ok(())
    }

    /// Attention + o_proj with persistent GPU KV cache.
    /// Q is uploaded from CPU; K/V are read from persistent per-layer GPU buffers.
    pub fn attention_and_proj_i4_persistent_kv(
        &self,
        q: &[f32],
        k_buf: &Buffer,
        v_buf: &Buffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        attn_scale: f32,
        soft_cap: f32,
        o_proj_w: &[u8],
        o_proj_s: &[u16],
        o_proj_out_dim: usize,
        o_proj_in_dim: usize,
        o_proj_key: &str,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        assert_eq!(q.len(), n_heads * head_dim);
        assert_eq!(out.len(), o_proj_out_dim);

        let q_bytes = (q.len() * 4) as u64;
        self.ensure_named_buf("attn_q", q.len())?;

        // 1) Upload Q
        let mut tmp_q: Option<Buffer> = None;
        {
            let nb = self.named_bufs.lock().unwrap();
            let b = nb.get("attn_q").unwrap();
            if b.length() < q_bytes {
                tmp_q = Some(upload_f32(&self.device, q)?);
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        q.as_ptr() as *const u8,
                        b.contents() as *mut u8,
                        q_bytes as usize,
                    );
                }
            }
        }

        // 2) Dispatch attention (fast kernel, reads K/V from persistent buffers)
        let q_buf_ref;
        {
            let nb = self.named_bufs.lock().unwrap();
            q_buf_ref = nb.get("attn_q").cloned().unwrap();
        }
        let q_buf = tmp_q.as_ref().unwrap_or(&q_buf_ref);
        let attn_out_name = "attn_out_persistent";
        self.ensure_named_buf(attn_out_name, n_heads * head_dim)?;
        let attn_out_buf;
        {
            let nb = self.named_bufs.lock().unwrap();
            attn_out_buf = nb.get(attn_out_name).cloned().unwrap();
        }

        let cmd_buf = self.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        self.encode_attention_gqa_f32(
            &enc,
            q_buf,
            k_buf,
            v_buf,
            &attn_out_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            attn_scale,
            soft_cap,
            kv_stride,
        );
        enc.end_encoding();

        // 3) o_proj (i4)
        let o_proj_name = format!("gemma.o_proj.{}", o_proj_key);
        let w_name = format!("o_proj_w.{}", o_proj_key);
        let s_name = format!("o_proj_s.{}", o_proj_key);
        let mut cache = self.tensor_cache.lock().unwrap();
        if !cache.contains_key(&w_name) {
            cache.insert(w_name.clone(), upload_i8(&self.device, unsafe { std::slice::from_raw_parts(o_proj_w.as_ptr() as *const i8, o_proj_w.len()) })?);
        }
        if !cache.contains_key(&s_name) {
            cache.insert(s_name.clone(), upload_u16(&self.device, o_proj_s)?);
        }
        drop(cache);
        let wbuf = self.tensor_cache.lock().unwrap().get(&w_name).cloned().unwrap();
        let sbuf = self.tensor_cache.lock().unwrap().get(&s_name).cloned().unwrap();
        let proj_out_name = "attn_proj_out_persistent";
        self.ensure_named_buf(proj_out_name, o_proj_out_dim)?;

        let enc2 = cmd_buf.new_compute_command_encoder();
        let proj_out_buf = self.ensure_named_buf(proj_out_name, 1).ok().and_then(|_| {
            self.named_bufs.lock().unwrap().get(proj_out_name).cloned()
        }).unwrap_or_else(|| attn_out_buf.clone());
        self.encode_mv_i4(
            &enc2,
            &wbuf,
            &sbuf,
            &attn_out_buf,
            &proj_out_buf,
            o_proj_out_dim,
            o_proj_in_dim,
            o_proj_in_dim,
        );
        enc2.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        self.read_named_buf(proj_out_name, out)?;
        Ok(())
    }

    /// Batch multiple i4 matrix-vector multiplications that share the same input vector.
    /// Each job = (w_u8, s_f16, out_dim, in_dim, gs, cache_key).
    /// Output slices are passed separately by index to satisfy the borrow checker.
    pub fn batch_mv_i4(
        &self,
        x: &[f32],
        jobs: &[(&[u8], &[u16], usize, usize, usize, &str)],
        outs: &mut [&mut [f32]],
    ) -> anyhow::Result<()> {
        assert_eq!(jobs.len(), outs.len(), "batch_mv_i4: job count must match output count");
        if jobs.is_empty() {
            return Ok(());
        }

        // 1. Cache all weights and scales.
        let mut wbuffers = Vec::with_capacity(jobs.len());
        let mut sbuffers = Vec::with_capacity(jobs.len());
        for (w_u8, s_f16, _out_dim, _in_dim, _gs, cache_key) in jobs.iter() {
            let w_key = format!("logits.w.{}", cache_key);
            let s_key = format!("logits.s.{}", cache_key);
            {
                let mut cache = self.tensor_cache.lock().unwrap();
                if !cache.contains_key(&w_key) {
                    cache.insert(w_key.clone(), upload_i8(&self.device, unsafe {
                        std::slice::from_raw_parts(w_u8.as_ptr() as *const i8, w_u8.len())
                    })?);
                }
                if !cache.contains_key(&s_key) {
                    cache.insert(s_key.clone(), upload_u16(&self.device, s_f16)?);
                }
                wbuffers.push(cache.get(&w_key).unwrap().clone());
                sbuffers.push(cache.get(&s_key).unwrap().clone());
            }
        }

        // 2. Ensure input buffer and named output buffers.
        ensure_buf_f32(&self.device, &mut *self.x_buf.lock().unwrap(), x.len())?;
        for (idx, (out_dim, ..)) in jobs.iter().map(|j| (j.2, j.3, j.4, j.5)).enumerate() {
            let out_name = format!("i4_batch_out_{}", idx);
            self.ensure_named_buf(&out_name, out_dim)?;
        }

        // 3. Write input.
        {
            let xb_ref = self.x_buf.lock().unwrap();
            let xb = xb_ref.as_ref().unwrap();
            write_f32(xb, x)?;
        }

        // 4. Collect output buffer handles.
        let mut out_buffers = Vec::with_capacity(jobs.len());
        for idx in 0..jobs.len() {
            let out_name = format!("i4_batch_out_{}", idx);
            out_buffers.push(self.get_named_buf(&out_name)?);
        }

        // 5. Run all kernels in a single command buffer.
        let xb_guard = self.x_buf.lock().unwrap();
        let xb = xb_guard.as_ref().unwrap();
        self.run_batch(|enc| {
            for (idx, job) in jobs.iter().enumerate() {
                let out_dim = job.2;
                let in_dim = job.3;
                let gs = job.4;
                self.encode_mv_i4(enc, &wbuffers[idx], &sbuffers[idx], xb, &out_buffers[idx], out_dim, in_dim, gs);
            }
            Ok(())
        })?;
        drop(xb_guard);

        // 6. Read all outputs.
        for (idx, out) in outs.iter_mut().enumerate() {
            let out_name = format!("i4_batch_out_{}", idx);
            self.read_named_buf(&out_name, out)?;
        }

        Ok(())
    }

    /// Fused MLP block for Gemma: gate_proj + up_proj + gelu_tanh_mul + down_proj +
    /// optional post-FFN RMSNorm + residual_add, all in a single command buffer.
    /// This saves the intermediate CPU↔GPU round-trips that occur when gate/up are
    /// read back for CPU GELU and then re-uploaded for down_proj.
    pub fn fused_mlp_i4(
        &self,
        mlp_in: &[f32],
        residual: &[f32],
        gate_w: &[u8],
        gate_s: &[u16],
        gate_dim: usize,
        gate_key: &str,
        up_w: &[u8],
        up_s: &[u16],
        up_dim: usize,
        up_key: &str,
        down_w: &[u8],
        down_s: &[u16],
        down_out_dim: usize,
        down_key: &str,
        post_norm_w_f16: Option<&[u16]>,
        post_norm_key: Option<&str>,
        post_norm_eps: f32,
        post_norm_add_one: bool,
        hidden: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        assert_eq!(mlp_in.len(), hidden);
        assert_eq!(residual.len(), hidden);
        assert_eq!(out.len(), hidden);
        assert_eq!(gate_dim, up_dim);

        // 1. Cache weights and scales.
        let gate_w_key = format!("mlp.gate.w.{}", gate_key);
        let gate_s_key = format!("mlp.gate.s.{}", gate_key);
        let up_w_key = format!("mlp.up.w.{}", up_key);
        let up_s_key = format!("mlp.up.s.{}", up_key);
        let down_w_key = format!("mlp.down.w.{}", down_key);
        let down_s_key = format!("mlp.down.s.{}", down_key);

        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&gate_w_key) {
                cache.insert(gate_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(gate_w.as_ptr() as *const i8, gate_w.len())
                })?);
            }
            if !cache.contains_key(&gate_s_key) {
                cache.insert(gate_s_key.clone(), upload_u16(&self.device, gate_s)?);
            }
            if !cache.contains_key(&up_w_key) {
                cache.insert(up_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(up_w.as_ptr() as *const i8, up_w.len())
                })?);
            }
            if !cache.contains_key(&up_s_key) {
                cache.insert(up_s_key.clone(), upload_u16(&self.device, up_s)?);
            }
            if !cache.contains_key(&down_w_key) {
                cache.insert(down_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(down_w.as_ptr() as *const i8, down_w.len())
                })?);
            }
            if !cache.contains_key(&down_s_key) {
                cache.insert(down_s_key.clone(), upload_u16(&self.device, down_s)?);
            }
        }

        // Cache post-FFN norm weight if provided.
        let post_norm_w_key = post_norm_key.map(|k| format!("mlp.post_norm.w.{}", k));
        if let (Some(w), Some(key)) = (post_norm_w_f16, post_norm_key) {
            let w_key = format!("mlp.post_norm.w.{}", key);
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&w_key) {
                cache.insert(w_key.clone(), upload_u16(&self.device, w)?);
            }
        }

        // 2. Ensure named buffers.
        self.ensure_named_buf("mlp_in", hidden)?;
        self.ensure_named_buf("mlp_residual", hidden)?;
        self.ensure_named_buf("mlp_gate", gate_dim)?;
        self.ensure_named_buf("mlp_up", up_dim)?;
        self.ensure_named_buf("mlp_down", down_out_dim)?;
        if post_norm_w_f16.is_some() {
            self.ensure_named_buf("mlp_down_normed", down_out_dim)?;
        }

        // 3. Upload inputs.
        self.write_named_buf("mlp_in", mlp_in)?;
        self.write_named_buf("mlp_residual", residual)?;

        // 4. Get buffer handles.
        let cache = self.tensor_cache.lock().unwrap();
        let gate_wb = cache.get(&gate_w_key).unwrap().clone();
        let gate_sb = cache.get(&gate_s_key).unwrap().clone();
        let up_wb = cache.get(&up_w_key).unwrap().clone();
        let up_sb = cache.get(&up_s_key).unwrap().clone();
        let down_wb = cache.get(&down_w_key).unwrap().clone();
        let down_sb = cache.get(&down_s_key).unwrap().clone();
        let post_norm_wb = post_norm_w_key.as_ref().and_then(|k| cache.get(k).cloned());
        drop(cache);

        let mlp_in_buf = self.get_named_buf("mlp_in")?;
        let residual_buf = self.get_named_buf("mlp_residual")?;
        let gate_buf = self.get_named_buf("mlp_gate")?;
        let up_buf = self.get_named_buf("mlp_up")?;
        let down_buf = self.get_named_buf("mlp_down")?;
        let down_normed_buf = if post_norm_w_f16.is_some() {
            Some(self.get_named_buf("mlp_down_normed")?)
        } else {
            None
        };

        // 5. Encode everything in one command buffer.
        self.run_batch(|enc| {
            // gate = mlp_in @ W_gate
            self.encode_mv_i4(enc, &gate_wb, &gate_sb, &mlp_in_buf, &gate_buf, gate_dim, hidden, hidden);
            // up = mlp_in @ W_up
            self.encode_mv_i4(enc, &up_wb, &up_sb, &mlp_in_buf, &up_buf, up_dim, hidden, hidden);
            // gate = GELU(gate) * up  (in-place on gate)
            self.encode_gelu_tanh_mul_f32_inplace(enc, &gate_buf, &up_buf, gate_dim);
            // down = gate @ W_down  (gs = gate_dim because input to down_proj is gate_dim/ ffn_dim)
            self.encode_mv_i4(enc, &down_wb, &down_sb, &gate_buf, &down_buf, down_out_dim, gate_dim, gate_dim);
            // Optional post-FFN RMSNorm, then residual add.
            if let (Some(wb), Some(dnb)) = (&post_norm_wb, &down_normed_buf) {
                self.encode_rms_norm_f16w(enc, &down_buf, wb, dnb, down_out_dim, post_norm_eps, post_norm_add_one);
                self.encode_add_f32_inplace(enc, &residual_buf, dnb, down_out_dim);
            } else {
                self.encode_add_f32_inplace(enc, &residual_buf, &down_buf, down_out_dim);
            }
            Ok(())
        })?;

        // 6. Read result back.
        self.read_named_buf("mlp_residual", out)?;
        Ok(())
    }

    /// Fused block: post-attention norm + residual + pre-FFN norm + MLP (gate+up+gelu+down+
    /// optional post-FFN norm + residual), all in one command buffer.
    /// This eliminates the CPU↔GPU round-trips between o_proj and the MLP.
    pub fn fused_post_attn_residual_mlp_i4(
        &self,
        attn_proj: &[f32],
        x: &[f32],
        post_attn_norm_w: &[u16],
        post_attn_norm_eps: f32,
        post_attn_norm_add_one: bool,
        pre_ffn_norm_w: &[u16],
        pre_ffn_norm_eps: f32,
        pre_ffn_norm_add_one: bool,
        gate_w: &[u8],
        gate_s: &[u16],
        gate_dim: usize,
        gate_key: &str,
        up_w: &[u8],
        up_s: &[u16],
        up_dim: usize,
        up_key: &str,
        down_w: &[u8],
        down_s: &[u16],
        down_out_dim: usize,
        down_key: &str,
        post_ffn_norm_w: Option<&[u16]>,
        post_ffn_norm_key: Option<&str>,
        post_ffn_norm_eps: f32,
        post_ffn_norm_add_one: bool,
        hidden: usize,
        out: &mut [f32],
    ) -> anyhow::Result<()> {
        assert_eq!(attn_proj.len(), hidden);
        assert_eq!(x.len(), hidden);
        assert_eq!(out.len(), hidden);
        assert_eq!(gate_dim, up_dim);

        // 1. Cache all weights.
        let pa_key = format!("tail.pa.w.{gate_key}");
        let pf_key = format!("tail.pf.w.{gate_key}");
        let gate_w_key = format!("tail.gate.w.{gate_key}");
        let gate_s_key = format!("tail.gate.s.{gate_key}");
        let up_w_key = format!("tail.up.w.{gate_key}");
        let up_s_key = format!("tail.up.s.{gate_key}");
        let down_w_key = format!("tail.down.w.{gate_key}");
        let down_s_key = format!("tail.down.s.{gate_key}");

        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&pa_key) {
                cache.insert(pa_key.clone(), upload_u16(&self.device, post_attn_norm_w)?);
            }
            if !cache.contains_key(&pf_key) {
                cache.insert(pf_key.clone(), upload_u16(&self.device, pre_ffn_norm_w)?);
            }
            if !cache.contains_key(&gate_w_key) {
                cache.insert(gate_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(gate_w.as_ptr() as *const i8, gate_w.len())
                })?);
            }
            if !cache.contains_key(&gate_s_key) {
                cache.insert(gate_s_key.clone(), upload_u16(&self.device, gate_s)?);
            }
            if !cache.contains_key(&up_w_key) {
                cache.insert(up_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(up_w.as_ptr() as *const i8, up_w.len())
                })?);
            }
            if !cache.contains_key(&up_s_key) {
                cache.insert(up_s_key.clone(), upload_u16(&self.device, up_s)?);
            }
            if !cache.contains_key(&down_w_key) {
                cache.insert(down_w_key.clone(), upload_i8(&self.device, unsafe {
                    std::slice::from_raw_parts(down_w.as_ptr() as *const i8, down_w.len())
                })?);
            }
            if !cache.contains_key(&down_s_key) {
                cache.insert(down_s_key.clone(), upload_u16(&self.device, down_s)?);
            }
        }

        let post_ffn_key = post_ffn_norm_key.map(|k| format!("tail.post.w.{k}"));
        if let (Some(w), Some(key)) = (post_ffn_norm_w, post_ffn_norm_key) {
            let k = format!("tail.post.w.{key}");
            let mut cache = self.tensor_cache.lock().unwrap();
            if !cache.contains_key(&k) {
                cache.insert(k.clone(), upload_u16(&self.device, w)?);
            }
        }

        // 2. Ensure buffers.
        self.ensure_named_buf("tail_attn_proj", hidden)?;
        self.ensure_named_buf("tail_x", hidden)?;
        self.ensure_named_buf("tail_post_attn", hidden)?;
        self.ensure_named_buf("tail_mlp_in", hidden)?;
        self.ensure_named_buf("tail_gate", gate_dim)?;
        self.ensure_named_buf("tail_up", up_dim)?;
        self.ensure_named_buf("tail_down", hidden)?;
        if post_ffn_norm_w.is_some() {
            self.ensure_named_buf("tail_down_normed", hidden)?;
        }

        // 3. Upload inputs.
        self.write_named_buf("tail_attn_proj", attn_proj)?;
        self.write_named_buf("tail_x", x)?;

        // 4. Get handles.
        let cache = self.tensor_cache.lock().unwrap();
        let pa_wb = cache.get(&pa_key).unwrap().clone();
        let pf_wb = cache.get(&pf_key).unwrap().clone();
        let gate_wb = cache.get(&gate_w_key).unwrap().clone();
        let gate_sb = cache.get(&gate_s_key).unwrap().clone();
        let up_wb = cache.get(&up_w_key).unwrap().clone();
        let up_sb = cache.get(&up_s_key).unwrap().clone();
        let down_wb = cache.get(&down_w_key).unwrap().clone();
        let down_sb = cache.get(&down_s_key).unwrap().clone();
        let post_ffn_wb = post_ffn_key.as_ref().and_then(|k| cache.get(k).cloned());
        drop(cache);

        let attn_proj_buf = self.get_named_buf("tail_attn_proj")?;
        let x_buf = self.get_named_buf("tail_x")?;
        let post_attn_buf = self.get_named_buf("tail_post_attn")?;
        let mlp_in_buf = self.get_named_buf("tail_mlp_in")?;
        let gate_buf = self.get_named_buf("tail_gate")?;
        let up_buf = self.get_named_buf("tail_up")?;
        let down_buf = self.get_named_buf("tail_down")?;
        let down_normed_buf = if post_ffn_norm_w.is_some() {
            Some(self.get_named_buf("tail_down_normed")?)
        } else {
            None
        };

        // 5. Encode: post-attn norm → residual → pre-FFN norm → gate → up → gelu → down → post-FFN norm (opt) → residual.
        self.run_batch(|enc| {
            // post_attn_normed = RMSNorm(attn_proj)
            self.encode_rms_norm_f16w(enc, &attn_proj_buf, &pa_wb, &post_attn_buf, hidden, post_attn_norm_eps, post_attn_norm_add_one);
            // x += post_attn_normed   (now x = x + post_attn_norm(attn_proj))
            self.encode_add_f32_inplace(enc, &x_buf, &post_attn_buf, hidden);
            // mlp_in = RMSNorm(x)
            self.encode_rms_norm_f16w(enc, &x_buf, &pf_wb, &mlp_in_buf, hidden, pre_ffn_norm_eps, pre_ffn_norm_add_one);
            // gate = mlp_in @ W_gate
            self.encode_mv_i4(enc, &gate_wb, &gate_sb, &mlp_in_buf, &gate_buf, gate_dim, hidden, hidden);
            // up = mlp_in @ W_up
            self.encode_mv_i4(enc, &up_wb, &up_sb, &mlp_in_buf, &up_buf, up_dim, hidden, hidden);
            // gate = GELU(gate) * up  (in-place on gate)
            self.encode_gelu_tanh_mul_f32_inplace(enc, &gate_buf, &up_buf, gate_dim);
            // down = gate @ W_down
            self.encode_mv_i4(enc, &down_wb, &down_sb, &gate_buf, &down_buf, down_out_dim, gate_dim, gate_dim);
            // optional post-FFN norm, then residual add into x_buf
            if let (Some(pwb), Some(dnb)) = (&post_ffn_wb, &down_normed_buf) {
                self.encode_rms_norm_f16w(enc, &down_buf, pwb, dnb, down_out_dim, post_ffn_norm_eps, post_ffn_norm_add_one);
                self.encode_add_f32_inplace(enc, &x_buf, dnb, down_out_dim);
            } else {
                self.encode_add_f32_inplace(enc, &x_buf, &down_buf, down_out_dim);
            }
            Ok(())
        })?;

        // 6. Read x back (final hidden state for this layer).
        self.read_named_buf("tail_x", out)?;
        Ok(())
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pipeline(device: &Device, src: &str, fn_name: &str) -> anyhow::Result<(Library, ComputePipelineState)> {
    let lib = load_or_compile_metallib(device, src, false, &format!("matmul_{}", fn_name))?;
    let func = lib.get_function(fn_name, None).map_err(|e| anyhow::anyhow!("Missing fn {fn_name}: {e:?}"))?;
    let pso = device.new_compute_pipeline_state_with_function(&func).map_err(|e| anyhow::anyhow!("PSO failed: {e:?}"))?;
    Ok((lib, pso))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_pso_ops(device: &metal::DeviceRef, lib: &Library, name: &str) -> anyhow::Result<ComputePipelineState> {
    let mut cache_guard = PSO_CACHE.lock().unwrap();
    if cache_guard.is_none() {
        *cache_guard = Some(HashMap::new());
    }

    let cache = cache_guard.as_mut().unwrap();
    if let Some(pso) = cache.get(name) {
        return Ok(pso.clone());
    }

    let f = lib.get_function(name, None).map_err(|_| anyhow::anyhow!("Missing function {name}"))?;
    let pso = device.new_compute_pipeline_state_with_function(&f).map_err(|e| anyhow::anyhow!("PSO {name} failed: {e:?}"))?;
    cache.insert(name.to_string(), pso.clone());
    Ok(pso)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_f32(device: &metal::DeviceRef, src: &[f32]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer((src.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut f32; if ptr.is_null() { anyhow::bail!("Upload f32 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_u16(device: &metal::DeviceRef, src: &[u16]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer((src.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut u16; if ptr.is_null() { anyhow::bail!("Upload u16 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn upload_i8(device: &metal::DeviceRef, src: &[i8]) -> anyhow::Result<Buffer> {
    let b = device.new_buffer(src.len() as u64, MTLResourceOptions::StorageModeShared);
    let ptr = b.contents() as *mut i8; if ptr.is_null() { anyhow::bail!("Upload i8 failed"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(b)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ensure_buf_f32(device: &metal::DeviceRef, slot: &mut Option<Buffer>, elems: usize) -> anyhow::Result<()> {
    let need = (elems * 4) as u64;
    if slot.as_ref().map(|b| b.length() >= need).unwrap_or(false) { return Ok(()); }
    let b = device.new_buffer(need, MTLResourceOptions::StorageModeShared);
    if b.contents().is_null() { anyhow::bail!("Scratch f32 failed"); }
    *slot = Some(b); Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_f32(buf: &Buffer, src: &[f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *mut f32; if ptr.is_null() { anyhow::bail!("Write f32 null"); }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn read_f32(buf: &Buffer, dst: &mut [f32]) -> anyhow::Result<()> {
    let ptr = buf.contents() as *const f32; if ptr.is_null() { anyhow::bail!("Read f32 null"); }
    unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()); }
    Ok(())
}
