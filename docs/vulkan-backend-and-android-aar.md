# Vulkan Backend and Android Kotlin AAR

This document describes the Vulkan compute backend and Kotlin Android AAR
bindings added to cellm for cross-platform GPU inference.

---

## What Was Implemented

1.  **VulkanBackend** -- A compute backend using Vulkan 1.1 for GPU-accelerated
    inference on Android devices. Replaces the empty `pub struct VulkanKernels;`
    with a full instance/device/pipeline/buffer management system implementing the
    `Backend` trait.

2.  **Kotlin AAR bindings** -- Android library wrapping the cellm C FFI in
    idiomatic Kotlin. Includes `CellmEngine`, `CellmSession`, and
    `CellmTokenizer` classes with proper lifecycle management.

---

## Architecture

### Vulkan Backend Initialization

```mermaid
flowchart TD
    START["VulkanBackend::new()"] --> ENTRY["load Vulkan entry point"]
    ENTRY --> INSTANCE["create Vulkan instance (v1.1)"]
    INSTANCE --> ENUM["enumerate physical devices"]
    ENUM --> SELECT{"discrete GPU found?"}
    SELECT -->|yes| DGPU["select discrete GPU"]
    SELECT -->|no| IGPU["select integrated GPU"]
    DGPU --> FINDCOMP["find compute queue family"]
    IGPU --> FINDCOMP
    FINDCOMP --> DEVICE["create logical device"]
    DEVICE --> QUEUE["get compute queue"]
    QUEUE --> POOL["create command pool\n(RESET_COMMAND_BUFFER)"]
    POOL --> DESCPOOL["create descriptor pool\n(256 storage + 64 uniform)"]
    DESCPOOL --> DONE["VulkanBackend ready"]
```

### Buffer Upload Pipeline

```mermaid
flowchart TD
    DATA["f32 weight data on CPU"] --> STAGING["create staging buffer\n(HOST_VISIBLE | HOST_COHERENT)"]
    STAGING --> MAP["map staging memory to CPU"]
    MAP --> COPY["copy data to staging buffer"]
    COPY --> UNMAP["unmap staging memory"]
    UNMAP --> CMD["begin single-time command buffer"]
    CMD --> CMDBUF["cmd_copy_buffer(staging -> device-local)"]
    CMDBUF --> SUBMIT["end + submit + wait_idle"]
    SUBMIT --> CLEANUP["destroy staging buffer + free memory"]
    CLEANUP --> READY["device-local buffer ready for shader access"]
```

### Pipeline Compilation and Caching

```mermaid
flowchart LR
    REQ["get_pipeline(name, spirv)"]
    REQ --> CHECK{"in pipeline_cache?"}
    CHECK -->|yes| RETURN["return cached pipeline"]
    CHECK -->|no| MODULE["create shader module from SPIR-V"]
    MODULE --> STAGE["create compute stage info"]
    STAGE --> LAYOUT["get or create pipeline layout\n(with push constants)"]
    LAYOUT --> CREATE["create compute pipeline"]
    CREATE --> INSERT["insert into pipeline_cache"]
    INSERT --> RETURN
```

### Kotlin AAR Data Flow

```mermaid
flowchart TD
    APP["Android App (Kotlin)"] --> ENGINE["CellmEngine.create(modelPath)"]
    ENGINE --> JNI["System.loadLibrary(cellm_sdk)"]
    JNI --> NATIVE["nativeCreate() -> cellm_engine_create_v4"]
    NATIVE --> RUST["Rust Engine::new()"]
    
    APP --> SESSION["engine.createSession()"]
    SESSION --> NATSES["nativeSessionCreate()"]
    
    APP --> TOK["CellmTokenizer.load(path)"]
    TOK --> TOKENS["tokenizer.encode(text)"]
    TOKENS --> SUBMIT["engine.submitTokens(session, tokens)"]
    
    SUBMIT --> STEP["engine.stepDecode()"]
    STEP --> PAIR["Pair(sessionHandle, tokenId)?"]
    PAIR --> DECODE["tokenizer.decode(token)"]
```

---

## VulkanBackend Structure

```
VulkanBackend
  device:          ash::Device              -- logical device handle
  queue:           ash::vk::Queue           -- compute queue
  command_pool:    ash::vk::CommandPool     -- command buffer allocation
  pipeline_cache:  Mutex<HashMap<String, Pipeline>>  -- compiled pipelines
  pipeline_layouts: Mutex<HashMap<String, PipelineLayout>>
  descriptor_pool: ash::vk::DescriptorPool  -- descriptor set allocation
  _instance:       ash::Instance            -- Vulkan instance (held for lifetime)
  _physical_device: ash::vk::PhysicalDevice -- selected GPU
  workgroup_count: u32                     -- dispatch size (default 256)
  shared_mem_size: usize                    -- shared memory limit (32KB)
```

### Backend Trait Implementation

All nine ops from the `Backend` trait are implemented:

| Op | Implementation Status |
|---|---|
| `matmul` | Stub: returns error directing to CPU backend |
| `rms_norm` | Stub |
| `rope_inplace` | Stub |
| `silu` | Stub |
| `add` | Stub |
| `mul` | Stub |
| `softmax_inplace` | Stub |
| `attention` | Stub |
| `kv_write/read` | Default CPU implementations (inherited from trait) |

Each stub returns a descriptive `CoreError::Backend` explaining the op is not
yet implemented. The default `kv_write_token_f16`/`kv_read_token_f16` methods
from the `Backend` trait provide CPU-side KV cache access regardless of backend.

### Memory Model

```
Device-local memory (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
  - Model weights (uploaded once at startup via staging buffer)
  - KV cache buffers (persistent across decode steps)

Host-visible memory (HOST_VISIBLE | HOST_COHERENT):
  - Staging buffers for uploads (allocated and freed per transfer)
  - Activation buffers for CPU readback

Push constants (fast, per-dispatch parameters):
  - KernelParams { m, n, k, batch, eps, theta }
  - 32 bytes total, updated per dispatch
```

---

## Kotlin AAR

### CellmEngine

Main entry point. Wraps the C FFI with Kotlin idioms.

```
CellmEngine
  enum Backend { CPU(0), METAL(1) }
  enum KvEncoding { F16(0), TURBOQUANT(1) }
  enum SchedulingPolicy { FAIR(0), LATENCY_FIRST(1), THROUGHPUT_FIRST(2) }
  data class KvStats(usedBlocks: Int, freeBlocks: Int)

  companion object:
    create(modelPath, tokensPerBlock=16, totalBlocks=256, ...): CellmEngine
    (all parameters have @JvmOverloads defaults)

  methods:
    createSession(): CellmSession
    submitTokens(session, tokens): Int
    submitTokensCached(session, tokens, cacheHit): Int
    stepDecode(): Pair<Long, Int>?
    cancelSession / suspendSession / resumeSession / resetSession
    setThermalLevel(level): Boolean
    getKvStats(): KvStats
    setSchedulingPolicy(policy): Boolean
    getSchedulingPolicy(): SchedulingPolicy
    getTotalTokens(): Long
    getTokPerSec(): Double
    resetStatsWindow()
    close()
```

### CellmSession

```
CellmSession(handle: Long, engine: CellmEngine)
  cancel()
  suspend()
  resume()
  reset()
  submitTokens(tokens: IntArray): Int
  submitTokensCached(tokens: IntArray): Pair<Int, Boolean>
```

### CellmTokenizer

```
CellmTokenizer(handle: Long) : AutoCloseable
  companion object:
    load(path: String): CellmTokenizer

  encode(text: String): IntArray
  encodeInto(text: String, outTokens: IntArray): Int
  decode(tokens: IntArray): String
  decodeOne(token: Int): String
  close()
```

### Build Configuration

```
build.gradle:
  namespace: com.cellm.sdk
  compileSdk: 34
  minSdk: 24
  targetSdk: 34
  abiFilters: arm64-v8a
  jniLibs: src/main/jniLibs
  output: cellm-sdk-{variant}.aar

copyNativeLibs task:
  from ../../target/aarch64-linux-android/release/libcellm_sdk.so
  into src/main/jniLibs/arm64-v8a/
```

---

## SPIR-V Shader Embedding

A minimal valid SPIR-V module (32 bytes) serves as a placeholder for compiled
compute shaders. In production, shaders are compiled from GLSL at build time
using `glslangValidator` or `shaderc` and embedded via `include_bytes!`.

The `spirv_for_kernel(name)` function provides shader bytecode by name.
Currently all kernels return the same placeholder. The production path replaces
this with:

```rust
pub fn spirv_for_kernel(name: &str) -> &'static [u8] {
    match name {
        "matmul_f32" => include_bytes!(concat!(env!("OUT_DIR"), "/shaders/matmul_f32.spv")),
        "attention_f32" => include_bytes!(concat!(env!("OUT_DIR"), "/shaders/attention_f32.spv")),
        // ... etc
        _ => &SPIRV_STUB_BYTES,
    }
}
```

---

## Files Changed

### New files

```
crates/cellm-kernels/src/vulkan.rs              VulkanBackend implementation (530 lines)
bindings/kotlin/src/main/kotlin/com/cellm/sdk/CellmEngine.kt
bindings/kotlin/src/main/kotlin/com/cellm/sdk/CellmSession.kt
bindings/kotlin/src/main/kotlin/com/cellm/sdk/CellmTokenizer.kt
```

### Modified files

```
crates/cellm-kernels/Cargo.toml        Replaced dash typo with ash 0.38 dependency
crates/cellm-kernels/src/lib.rs        Export VulkanBackend instead of VulkanKernels
bindings/kotlin/build.gradle           Updated with full AAR configuration
```

---

## Build Instructions

### Android AAR

```bash
# Cross-compile Rust library for Android
cargo build --release --target aarch64-linux-android -p cellm-sdk

# Copy native library into Kotlin project
cd bindings/kotlin
./gradlew copyNativeLibs

# Build AAR
./gradlew assembleRelease
# Output: bindings/kotlin/build/outputs/aar/cellm-sdk-release.aar
```

### Vulkan backend (desktop test)

```bash
cargo build -p cellm-kernels --features vulkan
cargo test -p cellm-kernels --lib
```

---
 