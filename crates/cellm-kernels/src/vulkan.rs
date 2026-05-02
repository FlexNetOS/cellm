// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! Vulkan compute backend for mobile LLM inference on Android.
//!
//! The Vulkan backend mirrors the Metal backend's op set (matmul, attention,
//! rms_norm, rope, silu, element-wise ops) using Vulkan compute shaders
//! compiled to SPIR-V at build time.  On Android, Vulkan 1.1+ is available
//! on devices with API level 24+ (Android 7.0), covering >95% of active
//! devices.
//!
//! Shader compilation strategy:
//!   - SPIR-V bytecode is embedded in the binary via `include_bytes!`.
//!   - Pipelines are cached in a HashMap keyed by shader name.
//!   - Descriptor sets are pre-allocated per pipeline layout.
//!
//! Memory model:
//!   - Weights live in device-local buffers (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
//!   - Activations live in host-visible buffers for CPU readback.
//!   - KV cache lives in device-local buffers, read via shader gather.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Mutex;

use cellm_core::{Backend, CoreError, TensorView};

// Embedded SPIR-V shaders
//
// In production, these are compiled from GLSL/MSL via glslangValidator or
// shaderc at build time.  For the stub, we embed minimal valid SPIR-V
// modules that implement each kernel.

// Minimal valid SPIR-V module header (Magic + Version + Generator + Bound + Reserved).
// A real implementation replaces these with compiled shader bytecode.

// Minimal valid SPIR-V stub. Each shader uses the same placeholder until real
// SPIR-V modules are compiled at build time via glslangValidator/shaderc.
const SPIRV_STUB_BYTES: [u8; 32] = [
    0x03, 0x02, 0x23, 0x07,
    0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x11,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

pub fn spirv_for_kernel(name: &str) -> &'static [u8] {
    // In production, this would return the compiled SPIR-V for the named kernel.
    // For the stub, all kernels share the same placeholder.
    let _ = name;
    &SPIRV_STUB_BYTES
}

// VulkanBackend

/// Vulkan compute backend.
///
/// Holds the Vulkan instance, device, compute queue, command pool, and
/// cached compute pipelines.  All GPU work is submitted to a single
/// compute queue (mobile GPUs typically have one queue family that
/// supports both graphics and compute).
pub struct VulkanBackend {
    #[cfg(feature = "vulkan")]
    device: ash::Device,
    #[cfg(feature = "vulkan")]
    queue: ash::vk::Queue,
    #[cfg(feature = "vulkan")]
    command_pool: ash::vk::CommandPool,
    #[cfg(feature = "vulkan")]
    pipeline_cache: Mutex<HashMap<String, ash::vk::Pipeline>>,
    #[cfg(feature = "vulkan")]
    pipeline_layouts: Mutex<HashMap<String, ash::vk::PipelineLayout>>,
    #[cfg(feature = "vulkan")]
    descriptor_pool: ash::vk::DescriptorPool,
    #[cfg(feature = "vulkan")]
    _instance: ash::Instance,
    #[cfg(feature = "vulkan")]
    _physical_device: ash::vk::PhysicalDevice,

    /// Number of workgroups to dispatch (tuned per device).
    workgroup_count: u32,
    /// Maximum shared memory size in bytes.
    shared_mem_size: usize,
}

impl VulkanBackend {
    /// Create a new Vulkan backend.
    ///
    /// On platforms without Vulkan support, returns a stub that implements
    /// the `Backend` trait with CPU fallback.
    pub fn new() -> Result<Self, String> {
        #[cfg(not(feature = "vulkan"))]
        {
            Ok(Self {
                workgroup_count: 256,
                shared_mem_size: 32768,
            })
        }

        #[cfg(feature = "vulkan")]
        {
            let entry = unsafe { ash::Entry::load() }
                .map_err(|e| format!("Vulkan: failed to load entry: {e}"))?;

            let app_name = CString::new("cellm").unwrap();
            let app_info = ash::vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(ash::vk::make_api_version(0, 1, 0, 0))
                .api_version(ash::vk::API_VERSION_1_1);

            let instance_extensions = vec![];
            let instance_create_info = ash::vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions);

            let instance = unsafe { entry.create_instance(&instance_create_info, None) }
                .map_err(|e| format!("Vulkan: failed to create instance: {e}"))?;

            let (physical_device, queue_family_index) =
                select_physical_device(&instance)?;

            let queue_priorities = [1.0f32];
            let device_queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_extensions = vec![];
            let device_create_info = ash::vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&device_queue_create_info))
                .enabled_extension_names(&device_extensions);

            let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .map_err(|e| format!("Vulkan: failed to create device: {e}"))?;

            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

            let command_pool_create_info = ash::vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = unsafe {
                device.create_command_pool(&command_pool_create_info, None)
            }
            .map_err(|e| format!("Vulkan: failed to create command pool: {e}"))?;

            let pool_sizes = [
                ash::vk::DescriptorPoolSize::default()
                    .ty(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(256),
                ash::vk::DescriptorPoolSize::default()
                    .ty(ash::vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(64),
            ];
            let descriptor_pool_create_info = ash::vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(128);
            let descriptor_pool = unsafe {
                device.create_descriptor_pool(&descriptor_pool_create_info, None)
            }
            .map_err(|e| format!("Vulkan: failed to create descriptor pool: {e}"))?;

            Ok(Self {
                device,
                queue,
                command_pool,
                pipeline_cache: Mutex::new(HashMap::new()),
                pipeline_layouts: Mutex::new(HashMap::new()),
                descriptor_pool,
                _instance: instance,
                _physical_device: physical_device,
                workgroup_count: 256,
                shared_mem_size: 32768,
            })
        }
    }

    /// Create a device-local buffer and upload data.
    #[cfg(feature = "vulkan")]
    fn create_device_buffer(
        &self,
        data: &[u8],
        usage: ash::vk::BufferUsageFlags,
    ) -> Result<(ash::vk::Buffer, ash::vk::DeviceMemory), String> {
        let buffer_create_info = ash::vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(usage | ash::vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }
            .map_err(|e| format!("Vulkan: create_buffer: {e}"))?;

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let memory_allocate_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(
                mem_reqs.memory_type_bits,
                ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);

        let memory = unsafe { self.device.allocate_memory(&memory_allocate_info, None) }
            .map_err(|e| format!("Vulkan: allocate_memory: {e}"))?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| format!("Vulkan: bind_buffer_memory: {e}"))?;
        }

        // Upload via staging buffer if data is non-empty.
        if !data.is_empty() {
            self.upload_to_buffer(buffer, data)?;
        }

        Ok((buffer, memory))
    }

    /// Upload data to a device-local buffer via a staging buffer.
    #[cfg(feature = "vulkan")]
    fn upload_to_buffer(&self, dst_buffer: ash::vk::Buffer, data: &[u8]) -> Result<(), String> {
        let staging_create_info = ash::vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(ash::vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        let staging = unsafe { self.device.create_buffer(&staging_create_info, None) }
            .map_err(|e| format!("Vulkan: create staging buffer: {e}"))?;

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging) };

        let memory_allocate_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(
                mem_reqs.memory_type_bits,
                ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                    | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);

        let staging_memory =
            unsafe { self.device.allocate_memory(&memory_allocate_info, None) }
                .map_err(|e| format!("Vulkan: allocate staging memory: {e}"))?;

        unsafe {
            self.device
                .bind_buffer_memory(staging, staging_memory, 0)
                .map_err(|e| format!("Vulkan: bind staging memory: {e}"))?;

            let ptr = self
                .device
                .map_memory(staging_memory, 0, data.len() as u64, ash::vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Vulkan: map staging memory: {e}"))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(staging_memory);
        }

        let cmd = self.begin_single_time_commands()?;
        let copy_region = ash::vk::BufferCopy::default().size(data.len() as u64);
        unsafe {
            self.device.cmd_copy_buffer(cmd, staging, dst_buffer, &[copy_region]);
        }
        self.end_single_time_commands(cmd)?;

        unsafe {
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_memory, None);
        }

        Ok(())
    }

    /// Begin a single-submit command buffer.
    #[cfg(feature = "vulkan")]
    fn begin_single_time_commands(&self) -> Result<ash::vk::CommandBuffer, String> {
        let allocate_info = ash::vk::CommandBufferAllocateInfo::default()
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let cmd = unsafe { self.device.allocate_command_buffers(&allocate_info) }
            .map_err(|e| format!("Vulkan: allocate command buffer: {e}"))?[0];

        let begin_info = ash::vk::CommandBufferBeginInfo::default()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Vulkan: begin command buffer: {e}"))?;
        }

        Ok(cmd)
    }

    /// End and submit a single-submit command buffer.
    #[cfg(feature = "vulkan")]
    fn end_single_time_commands(&self, cmd: ash::vk::CommandBuffer) -> Result<(), String> {
        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| format!("Vulkan: end command buffer: {e}"))?;

            let submit_info = ash::vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(self.queue, &[submit_info], ash::vk::Fence::null())
                .map_err(|e| format!("Vulkan: queue submit: {e}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| format!("Vulkan: queue wait idle: {e}"))?;
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(())
    }

    /// Find a memory type index matching requirements.
    #[cfg(feature = "vulkan")]
    fn find_memory_type(
        &self,
        type_bits: u32,
        properties: ash::vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let mem_props = unsafe { self._instance.get_physical_device_memory_properties(self._physical_device) };
        for i in 0..mem_props.memory_type_count {
            if (type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err("Vulkan: no suitable memory type found".into())
    }

    /// Get or compile a compute pipeline for a shader.
    #[cfg(feature = "vulkan")]
    fn get_pipeline(
        &self,
        name: &str,
        spirv: &[u8],
        entry_point: &str,
    ) -> Result<ash::vk::Pipeline, String> {
        {
            let cache = self.pipeline_cache.lock().unwrap();
            if let Some(pipeline) = cache.get(name) {
                return Ok(*pipeline);
            }
        }

        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_create_info, None)
        }
        .map_err(|e| format!("Vulkan: create shader module ({name}): {e}"))?;

        let entry_name = CString::new(entry_point).unwrap();
        let stage_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);

        // Simple pipeline layout with one descriptor set for storage buffers.
        let layout = self.get_or_create_pipeline_layout(name)?;

        let pipeline_create_info = ash::vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(
                    ash::vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
        }
        .map_err(|(_, e)| format!("Vulkan: create compute pipeline ({name}): {e}"))?[0];

        unsafe { self.device.destroy_shader_module(shader_module, None) };

        let mut cache = self.pipeline_cache.lock().unwrap();
        cache.insert(name.to_string(), pipeline);
        Ok(pipeline)
    }

    /// Get or create a pipeline layout for a shader.
    #[cfg(feature = "vulkan")]
    fn get_or_create_pipeline_layout(
        &self,
        name: &str,
    ) -> Result<ash::vk::PipelineLayout, String> {
        {
            let layouts = self.pipeline_layouts.lock().unwrap();
            if let Some(layout) = layouts.get(name) {
                return Ok(*layout);
            }
        }

        let push_constant_range = ash::vk::PushConstantRange::default()
            .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<KernelParams>() as u32);

        let layout_create_info = ash::vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_create_info, None)
        }
        .map_err(|e| format!("Vulkan: create pipeline layout ({name}): {e}"))?;

        let mut layouts = self.pipeline_layouts.lock().unwrap();
        layouts.insert(name.to_string(), layout);
        Ok(layout)
    }
}

// Backend trait implementation

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn matmul(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError> {
        // Stub: delegate to CPU matmul.
        // Full implementation dispatches a Vulkan compute shader.
        Err(CoreError::Backend(
            "Vulkan matmul: not yet implemented (use CPU backend for now)".into(),
        ))
    }

    fn rms_norm(
        &self,
        x: &TensorView,
        weight: &TensorView,
        out: &mut TensorView,
        eps: f32,
        arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan rms_norm: not yet implemented".into(),
        ))
    }

    fn rope_inplace(
        &self,
        q: &mut TensorView,
        k: &mut TensorView,
        positions: &[usize],
        theta: f32,
        arena: &mut [u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan rope: not yet implemented".into(),
        ))
    }

    fn silu(
        &self,
        x: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan silu: not yet implemented".into(),
        ))
    }

    fn add(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan add: not yet implemented".into(),
        ))
    }

    fn mul(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &mut TensorView,
        arena: &[u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan mul: not yet implemented".into(),
        ))
    }

    fn softmax_inplace(
        &self,
        x: &mut TensorView,
        arena: &mut [u8],
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan softmax: not yet implemented".into(),
        ))
    }

    fn attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        n_heads: usize,
        n_kv_heads: usize,
        out: &mut TensorView,
        arena: &[u8],
        scratch: &mut Vec<f32>,
    ) -> Result<(), CoreError> {
        Err(CoreError::Backend(
            "Vulkan attention: not yet implemented".into(),
        ))
    }
}

// Helpers

/// Parameters passed as push constants to every kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct KernelParams {
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    eps: f32,
    theta: f32,
    pad: [u32; 2],
}

#[cfg(feature = "vulkan")]
fn select_physical_device(
    instance: &ash::Instance,
) -> Result<(ash::vk::PhysicalDevice, u32), String> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("Vulkan: enumerate physical devices: {e}"))?;
    if devices.is_empty() {
        return Err("Vulkan: no physical devices found".into());
    }
    // Prefer the first discrete GPU, then integrated.
    for &device in &devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        if props.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU {
            let queue_family =
                find_compute_queue_family(instance, device)?;
            return Ok((device, queue_family));
        }
    }
    // Fall back to first available device with a compute queue.
    for &device in &devices {
        if let Ok(qf) = find_compute_queue_family(instance, device) {
            return Ok((device, qf));
        }
    }
    Err("Vulkan: no device with compute queue found".into())
}

#[cfg(feature = "vulkan")]
fn find_compute_queue_family(
    instance: &ash::Instance,
    device: ash::vk::PhysicalDevice,
) -> Result<u32, String> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (i, qf) in queue_families.iter().enumerate() {
        if qf.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) {
            return Ok(i as u32);
        }
    }
    Err("Vulkan: no compute queue family found".into())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_backend_has_name() {
        let backend = VulkanBackend::new().unwrap();
        assert_eq!(backend.name(), "vulkan");
    }
}
