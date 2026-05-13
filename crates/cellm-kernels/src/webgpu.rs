#![cfg(feature = "webgpu")]
// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! WebGPU compute-kernel backend for cellm.
//
// Provides GPU-accelerated matmul, RMS norm, RoPE, and softmax kernels
// via WGSL compute shaders dispatched through the `wgpu` crate.
// Targets `wasm32-unknown-unknown` with WebGPU and also works on native
// platforms (Vulkan/Metal/DX12) through wgpu's cross-platform layer.


// ---------------------------------------------------------------------------
// WGSL shader sources (embedded as constants)
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: out[m] = sum_k(A[m,k] * x[k])
/// A is stored row-major in buffer A; x is input vector; out is output.
const WGSL_MATMUL_F32: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,   // rows of A (output size)
    K: u32,   // cols of A (input size)
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.M { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        sum += A[row * params.K + k] * x[k];
    }
    out[row] = sum;
}
"#;

/// Matrix-vector multiply with f16 weight data (packed as u16).
/// A is [M × K] u16 elements (half-precision), x is [K] f32, out is [M] f32.
const WGSL_MATMUL_F16: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<u32>;  // packed: two u16 per u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

// Unpack f16 from u32 word (little-endian: lower 16 bits = first element)
fn unpack16(word: u32, idx: u32) -> f32 {
    let bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
    return bitcast<f16>(u16(bits));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.M { return; }
    var sum: f32 = 0.0;

    // Two u16 per u32 word; stride in u32 words = K / 2
    let stride = (params.K + 1u) / 2u;
    var k_even: u32 = 0u;
    for (var k: u32 = 0u; k < params.K; k += 2u) {
        let word = A[row * stride + k_even];
        let val0 = unpack16(word, 0u);
        sum += val0 * x[k];
        if k + 1u < params.K {
            let val1 = unpack16(word, 1u);
            sum += val1 * x[k + 1u];
        }
        k_even += 1u;
    }
    out[row] = sum;
}
"#;

/// RMS normalization: out[i] = x[i] * rsqrt(mean(x²) + eps) * weight[i]
const WGSL_RMS_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    N: u32,
    eps: f32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

// One workgroup computes RMS then each thread applies it.
// Workgroup reduction for mean-square calculation.
var<workgroup> partial_sq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;

    // Accumulate partial sum of squares
    var local_sq: f32 = 0.0;
    if i < params.N {
        let v = x[i];
        local_sq = v * v;
    }

    // Tree reduction within workgroup
    partial_sq[lid.x] = local_sq;
    workgroupBarrier();

    var step: u32 = 32u;
    while step > 0u {
        if lid.x < step {
            partial_sq[lid.x] += partial_sq[lid.x + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }

    let mean_sq = partial_sq[0] / f32(params.N);
    let rsqrt = 1.0 / sqrt(mean_sq + params.eps);

    // Apply normalization per thread
    if i < params.N {
        out[i] = x[i] * rsqrt * weight[i];
    }
}
"#;

/// RoPE (half rotary): apply RoPE to last `rotary_dim` elements of each head.
/// x layout: [n_heads, head_dim] interleaved f32. RoPE on last `rotary_dim` dims.
const WGSL_ROPE_HALF: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;

struct Uniforms {
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pos: u32,
    theta: f32,
};

@group(0) @binding(1) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x / params.rotary_dim;
    let i = gid.x % params.rotary_dim;
    if h >= params.n_heads || i >= params.rotary_dim / 2u { return; }

    // Index into this head's RoPE region
    let head_start = h * params.head_dim;
    let rope_start = head_start + params.head_dim - params.rotary_dim;

    let inv_freq = pow(params.theta, -f32(2u * i) / f32(params.rotary_dim));
    let angle = f32(params.pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let x0 = x[rope_start + i];
    let x1 = x[rope_start + params.rotary_dim / 2u + i];

    // Python convention: R(-θ) = [[cos, sin], [-sin, cos]]
    x[rope_start + i] = x0 * cos_val + x1 * sin_val;
    x[rope_start + params.rotary_dim / 2u + i] = -x0 * sin_val + x1 * cos_val;
}
"#;

/// Softmax: x[i] = exp(x[i] - max) / sum(exp(x[i] - max))
const WGSL_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;

struct Uniforms {
    N: u32,
};

@group(0) @binding(1) var<uniform> params: Uniforms;

var<workgroup> wg_max: f32;
var<workgroup> wg_sum: f32;
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let local_max: f32;

    // Pass 1: find max
    var local = -1e30f;
    if i < params.N { local = x[i]; }
    for (var j: u32 = 0u; j < 64u && i + j < params.N; j += 64u) {
        // Simple strided reduction — for small N (< 4096) this is fine
    }
    wg_max = local;  // Simplified — full workgroup reduction needed for >64 elems
    workgroupBarrier();

    let mx = wg_max;

    // Pass 2: compute exp sum
    var exp_sum: f32 = 0.0;
    if i < params.N {
        let v = exp(x[i] - mx);
        x[i] = v;
        exp_sum = v;
    }
    scratch[lid.x] = exp_sum;
    workgroupBarrier();

    // Reduction within workgroup
    var step: u32 = 32u;
    while step > 0u {
        if lid.x < step {
            scratch[lid.x] += scratch[lid.x + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }

    wg_sum = scratch[0];
    workgroupBarrier();

    // Pass 3: normalize
    if i < params.N {
        x[i] = x[i] / wg_sum;
    }
}
"#;

/// Silu-mul: gate[i] = silu(gate[i]) * up[i]
const WGSL_SILU_MUL: &str = r#"
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;

struct Uniforms {
    N: u32,
};

@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.N { return; }
    let g = gate[i];
    let s = 1.0 / (1.0 + exp(-g));
    gate[i] = (g * s) * up[i];
}
"#;

/// Batch matrix-vector multiply: out[r, o] = sum_k weight[o, k] * x[r, k]
/// weight is [M × K] row-major, x is [R × K] row-major, out is [R × M] row-major.
/// Each workgroup handles one output row (one row in x maps to one row in out).
const WGSL_MATMUL_BATCH_F32: &str = r#"
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,   // output cols (weight rows)
    K: u32,   // input cols  (weight cols)
    R: u32,   // number of input rows
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn batch_matmul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;      // output column (0..M)
    let row = gid.y;      // input row (0..R)
    if col >= params.M || row >= params.R { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k += 1u) {
        sum += weight[col * params.K + k] * x[row * params.K + k];
    }
    out[row * params.M + col] = sum;
}
"#;

/// Batch Layer Normalization: for each row, compute mean/variance, then normalize.
/// out[r, i] = (x[r, i] - mean(r)) * rsqrt(var(r) + eps) * weight[i] + bias[i]
const WGSL_LAYER_NORM_BATCH: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Uniforms {
    cols: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(4) var<uniform> params: Uniforms;

var<workgroup> wg_mean: f32;
var<workgroup> wg_var: f32;

@compute @workgroup_size(64)
fn batch_layer_norm(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.cols || row >= params.rows { return; }

    let base = row * params.cols;

    // Compute mean (single-thread per row is fine for small cols via loop)
    // Use workgroup reduction when cols <= 64 for the first lane per row
    var mean: f32 = 0.0;
    var var_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c += 1u) {
        let v = x[base + c];
        mean += v;
    }
    mean = mean / f32(params.cols);

    for (var c2: u32 = 0u; c2 < params.cols; c2 += 1u) {
        let d = x[base + c2] - mean;
        var_sum += d * d;
    }
    let inv = 1.0 / sqrt(var_sum / f32(params.cols) + params.eps);

    let v = x[base + col];
    out[base + col] = (v - mean) * inv * weight[col] + bias[col];
}
"#;

/// Batch RMS Normalization (per row).
/// out[r, i] = x[r, i] * rsqrt(mean(x[r]²) + eps) * weight[i]
const WGSL_RMS_NORM_BATCH: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    cols: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn batch_rms_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.cols || row >= params.rows { return; }

    let base = row * params.cols;

    var ms: f32 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c += 1u) {
        let v = x[base + c];
        ms += v * v;
    }
    ms = ms / f32(params.cols);
    let inv = 1.0 / sqrt(ms + params.eps);

    out[base + col] = x[base + col] * inv * weight[col];
}
"#;

// ---------------------------------------------------------------------------
// WebGPU backend state
// ---------------------------------------------------------------------------

/// Holds the WebGPU device, queue, and compiled shaders for inference.
pub struct WebGpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: WebGpuPipelines,
    limits: wgpu::Limits,
}

struct WebGpuPipelines {
    matmul_f32: wgpu::ComputePipeline,
    matmul_f16: wgpu::ComputePipeline,
    matmul_batch_f32: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    rms_norm_batch: wgpu::ComputePipeline,
    layer_norm_batch: wgpu::ComputePipeline,
    rope_half: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
}

impl WebGpuBackend {
    /// Create a WebGPU backend with pre-compiled shaders.
    /// Returns `None` if WebGPU is unavailable (falls back to CPU).
    pub async fn create() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cellm WebGPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256 MB
                        max_buffer_size: 256 * 1024 * 1024,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        let limits = device.limits();

        let pipelines = Self::build_pipelines(&device);

        Some(Self {
            device,
            queue,
            pipelines,
            limits,
        })
    }

    fn build_pipelines(device: &wgpu::Device) -> WebGpuPipelines {
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cellm-kernels-wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source().into()),
        });

        let make = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &sm,
                entry_point: Some(entry),
             compilation_options: Default::default(),
                cache: None,
            })
        };

        WebGpuPipelines {
            matmul_f32: make("matmul_f32"),
            matmul_f16: make("matmul_f16"),
            matmul_batch_f32: make("batch_matmul_f32"),
            rms_norm: make("rms_norm"),
            rms_norm_batch: make("batch_rms_norm"),
            layer_norm_batch: make("batch_layer_norm"),
            rope_half: make("rope_half"),
            softmax: make("softmax"),
            silu_mul: make("silu_mul"),
        }
    }

    // -----------------------------------------------------------------------
    // Upload helpers
    // -----------------------------------------------------------------------

    /// Upload f32 data to a GPU buffer with copy_dst + storage usage.
    pub fn upload_f32(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create an f32 storage buffer (uninitialized, for output).
    pub fn create_f32_output(&self, count: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload u16 data to GPU (packed as u32: two u16 per u32 for WGSL).
    pub fn upload_f16(&self, data: &[u16]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let n = data.len();
        let mut packed = Vec::with_capacity(n.max(1) / 2);
        for chunk in data.chunks(2) {
            let lo = chunk[0] as u32;
            let hi = if chunk.len() > 1 { (chunk[1] as u32) << 16 } else { 0 };
            packed.push(lo | hi);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Read back f32 results from a GPU buffer.
    pub fn download_f32(&self, buf: &wgpu::Buffer, count: usize, out: &mut [f32]) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&view));
        drop(view);
        staging.unmap();
    }

    // -----------------------------------------------------------------------
    // Kernel dispatch
    // -----------------------------------------------------------------------

    /// matmul f32: out[out_dim] = weight[out_dim × in_dim] · x[in_dim]
    /// Weight is on GPU (cached), x is uploaded fresh each call.
    pub fn matmul_f32(
        &self,
        w_buf: &wgpu::Buffer,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(out_dim as usize);

        let uniforms = [out_dim, in_dim, 0u32, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.matmul_f32.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_count = (out_dim + 63) / 64;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, out_dim as usize, out);
    }

    /// matmul f16: weight is stored as f16 (u16) on GPU.
    pub fn matmul_f16(
        &self,
        w_buf: &wgpu::Buffer,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(out_dim as usize);

        let uniforms = [out_dim, in_dim, 0u32, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.matmul_f16.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_f16);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((out_dim + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, out_dim as usize, out);
    }

    /// RMS norm: out[i] = x[i] * rsqrt(mean(x²) + eps) * weight[i]
    pub fn rms_norm(&self, weight_buf: &wgpu::Buffer, eps: f32, x: &[f32], out: &mut [f32]) {
        let n = x.len() as u32;
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(n as usize);

        let uniforms = [n, eps.to_bits(), 0u32, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rms_norm);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, n as usize, out);
    }

    /// RoPE half: apply RoPE to last `rotary_dim` elements of each head.
    pub fn rope_half(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        let x_buf = self.upload_f32(x);
        let uniforms = [
            n_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
            pos as u32,
            theta.to_bits(),
            0u32,
            0u32,
            0u32,
        ];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rope_half.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rope_half);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = n_heads as u32 * rotary_dim as u32;
            pass.dispatch_workgroups((total + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&x_buf, x.len(), x);
    }

    /// Softmax in-place.
    pub fn softmax(&self, data: &mut [f32]) {
        let n = data.len() as u32;
        let buf = self.upload_f32(data);
        let uniforms = [n, 0u32, 0u32, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.softmax.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.softmax);
            pass.set_bind_group(0, &bind_group, &[]);
            // Single workgroup for small softmax (< 64 elements)
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&buf, n as usize, data);
    }

    /// Silu-mul in-place: gate[i] = silu(gate[i]) * up[i]
    pub fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        let n = gate.len() as u32;
        let gate_buf = self.upload_f32(gate);
        let up_buf = self.upload_f32(up);
        let uniforms = [n, 0u32, 0u32, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.silu_mul.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gate_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: up_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.silu_mul);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&gate_buf, n as usize, gate);
    }

    // -----------------------------------------------------------------------
    // Batch kernels for vision encoder (multiple input rows per dispatch)
    // -----------------------------------------------------------------------

    /// Batch matmul: out[r, o] = sum_k weight[o, k] * x[r, k]
    /// weight is [out_dim × in_dim] row-major (cached on GPU in w_buf).
    /// x is [rows × in_dim] row-major, out is [rows × out_dim] row-major.
    pub fn matmul_batch_f32(
        &self,
        w_buf: &wgpu::Buffer,
        rows: u32,
        out_dim: u32,
        in_dim: u32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let total = rows * out_dim;
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output(total as usize);

        let uniforms = [out_dim, in_dim, rows, 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_batch_f32"),
                layout: &self.pipelines.matmul_batch_f32.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.matmul_batch_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (out_dim + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, total as usize, out);
    }

    /// Batch layer norm: out[r,i] = (x[r,i]-mean)*rsqrt(var+eps)*w[i]+b[i]
    pub fn layer_norm_batch(
        &self,
        w_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        rows: u32,
        cols: u32,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output((rows * cols) as usize);

        let uniforms = [cols, rows, eps.to_bits(), 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer_norm_batch"),
                layout: &self.pipelines.layer_norm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.layer_norm_batch);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (cols + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, (rows * cols) as usize, out);
    }

    /// Batch RMS norm: out[r,i] = x[r,i] * rsqrt(mean(x[r]²)+eps) * w[i]
    pub fn rms_norm_batch(
        &self,
        w_buf: &wgpu::Buffer,
        rows: u32,
        cols: u32,
        eps: f32,
        x: &[f32],
        out: &mut [f32],
    ) {
        let x_buf = self.upload_f32(x);
        let out_buf = self.create_f32_output((rows * cols) as usize);

        let uniforms = [cols, rows, eps.to_bits(), 0u32];
        let u_buf = self.upload_f32(bytemuck::cast_slice(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms_norm_batch"),
                layout: &self.pipelines.rms_norm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: u_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.rms_norm_batch);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (cols + 63) / 64;
            pass.dispatch_workgroups(wg_x, rows, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.download_f32(&out_buf, (rows * cols) as usize, out);
    }
}

// ---------------------------------------------------------------------------
// VisionWebGpu: high-level wrapper for vision-encoder GPU acceleration
// ---------------------------------------------------------------------------

/// Manages WebGPU weight caching and dispatch for vision-encoder linear
/// projections and normalizations.  Created from a `WebGpuBackend` and
/// passed into `vlm::run_vision_cellm` via `LinearBackend::WebGpu`.
///
/// Call `into_gpu()` after use to recover the `WebGpuBackend` for text inference.
pub struct VisionWebGpu {
    pub gpu: WebGpuBackend,
    weight_cache: std::collections::HashMap<(usize, usize, usize), wgpu::Buffer>,
}

impl VisionWebGpu {
    pub fn new(gpu: WebGpuBackend) -> Self {
        Self {
            gpu,
            weight_cache: std::collections::HashMap::new(),
        }
    }

    /// Recover the WebGpuBackend after vision processing is complete.
    pub fn into_gpu(self) -> WebGpuBackend {
        self.gpu
    }

    /// Upload a weight matrix (or retrieve cached) and run batch matmul.
    /// weight is [out_dim × in_dim] row-major; x is [rows × in_dim]; out is [rows × out_dim].
    pub fn linear_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        weight: &[f32],
        out_dim: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        if weight.is_empty() || rows == 0 || in_dim == 0 || out_dim == 0 {
            return;
        }

        let key = (weight.as_ptr() as usize, in_dim, out_dim);
        let w_buf = self.weight_cache.entry(key).or_insert_with(|| {
            // Store weight transposed: WGSL reads weight[col * K + k]
            // where weight is [M × K] with M=out_dim, K=in_dim.
            // Our input weight is [out_dim × in_dim] row-major, which
            // matches exactly — weight[col * in_dim + k] is correct.
            self.gpu.upload_f32(weight)
        });

        self.gpu.matmul_batch_f32(
            w_buf,
            rows as u32,
            out_dim as u32,
            in_dim as u32,
            x,
            out,
        );

        if let Some(b) = bias {
            for r in 0..rows {
                let row = &mut out[r * out_dim..(r + 1) * out_dim];
                for c in 0..out_dim {
                    row[c] += b[c];
                }
            }
        }
    }

    /// Upload weight+bias (or retrieve cached) and run batch layer norm.
    pub fn layer_norm_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        bias: &[f32],
        eps: f32,
        out: &mut [f32],
    ) {
        if cols == 0 || rows == 0 {
            return;
        }

        let w_key = (weight.as_ptr() as usize, cols, 0);
        let b_key = (bias.as_ptr() as usize, cols, 1);

        // Avoid double mutable borrow by checking existence first
        if !self.weight_cache.contains_key(&w_key) {
            self.weight_cache.insert(w_key, self.gpu.upload_f32(weight));
        }
        if !self.weight_cache.contains_key(&b_key) {
            self.weight_cache.insert(b_key, self.gpu.upload_f32(bias));
        }

        let w_buf = &self.weight_cache[&w_key];
        let b_buf = &self.weight_cache[&b_key];

        self.gpu.layer_norm_batch(
            w_buf, b_buf,
            rows as u32, cols as u32,
            eps, x, out,
        );
    }

    /// Upload weight (or retrieve cached) and run batch RMS norm.
    pub fn rms_norm_batch(
        &mut self,
        x: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) {
        if cols == 0 || rows == 0 {
            return;
        }

        let w_key = (weight.as_ptr() as usize, cols, 2);
        let w_buf = self.weight_cache.entry(w_key).or_insert_with(|| {
            self.gpu.upload_f32(weight)
        });

        self.gpu.rms_norm_batch(
            w_buf,
            rows as u32, cols as u32,
            eps, x, out,
        );
    }
}

// ---------------------------------------------------------------------------
// Combined WGSL source built at runtime
// ---------------------------------------------------------------------------

fn wgsl_source() -> String {
    let mut s = String::with_capacity(
        WGSL_MATMUL_F32.len() + WGSL_MATMUL_F16.len() + WGSL_MATMUL_BATCH_F32.len()
        + WGSL_RMS_NORM.len() + WGSL_RMS_NORM_BATCH.len() + WGSL_LAYER_NORM_BATCH.len()
        + WGSL_ROPE_HALF.len() + WGSL_SOFTMAX.len() + WGSL_SILU_MUL.len()
        + 64
    );
    s.push_str(WGSL_MATMUL_F32); s.push('\n');
    s.push_str(WGSL_MATMUL_F16); s.push('\n');
    s.push_str(WGSL_MATMUL_BATCH_F32); s.push('\n');
    s.push_str(WGSL_RMS_NORM); s.push('\n');
    s.push_str(WGSL_RMS_NORM_BATCH); s.push('\n');
    s.push_str(WGSL_LAYER_NORM_BATCH); s.push('\n');
    s.push_str(WGSL_ROPE_HALF); s.push('\n');
    s.push_str(WGSL_SOFTMAX); s.push('\n');
    s.push_str(WGSL_SILU_MUL);
    s
}
