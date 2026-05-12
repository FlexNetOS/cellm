// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! cellm-kernels: CPU SIMD, Metal compute shaders, WebGPU, and Vulkan.

pub mod cpu;
pub mod cpu_kernels;
pub mod metal;
pub mod vulkan;
pub mod wasm;
pub mod webgpu;

pub use cpu::SIMDKernels;
pub use metal::MetalKernels;
pub use metal::MetalOps;
pub use vulkan::VulkanBackend;
pub use webgpu::WebGpuBackend;
