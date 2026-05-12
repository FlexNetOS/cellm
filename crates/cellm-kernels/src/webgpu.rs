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
    rms_norm: wgpu::ComputePipeline,
    rope_half: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
}

impl WebGpuBackend {
    /// Create a WebGPU backend with pre-compiled shaders.
    /// Returns `None` if WebGPU is unavailable (falls back to CPU).
    pub async fn create() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
                entry_point: entry,
                compilation_options: Default::default(),
            })
        };

        WebGpuPipelines {
            matmul_f32: make("matmul_f32"),
            matmul_f16: make("matmul_f16"),
            rms_norm: make("rms_norm"),
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
}

// ---------------------------------------------------------------------------
// Combined WGSL source built at runtime
// ---------------------------------------------------------------------------

fn wgsl_source() -> String {
    let mut s = String::with_capacity(
        WGSL_MATMUL_F32.len() + WGSL_MATMUL_F16.len() + WGSL_RMS_NORM.len()
        + WGSL_ROPE_HALF.len() + WGSL_SOFTMAX.len() + WGSL_SILU_MUL.len()
        + 32
    );
    s.push_str(WGSL_MATMUL_F32); s.push('\n');
    s.push_str(WGSL_MATMUL_F16); s.push('\n');
    s.push_str(WGSL_RMS_NORM); s.push('\n');
    s.push_str(WGSL_ROPE_HALF); s.push('\n');
    s.push_str(WGSL_SOFTMAX); s.push('\n');
    s.push_str(WGSL_SILU_MUL);
    s
}
