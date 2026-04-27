# Q1_0_G128 CPU Optimization

<div style="text-align: right; font-size: 0.9em; color: gray;">April 26, 2026</div>


This document explains how we solved the severe CPU bottleneck when running extremely quantized models like Bonsai 1.7B.

## The Problem

When running the Bonsai model on the CPU fallback backend, inference was extremely slow. It took over 26 seconds to prefill 7 tokens and about 3 seconds per token to decode. 

We noticed that the CPU usage was stuck at exactly 103 percent. This indicated that the entire inference process was running on a single CPU core, failing to utilize the multi-core architecture of Apple Silicon devices.

Upon investigation, we found the root cause in the Qwen runner file. The Bonsai model is quantized to the `q1_0_g128` format, which is a 1.5 bit format with a group size of 128. For this specific data type, the CPU fallback path was executing a sequential loop over the output dimensions. For every row, it called a function that individually loaded tensor blocks and performed scalar bit extraction on a single thread. This forced the processor to compute billions of operations sequentially.

## The Solution

To fix this, we removed the sequential loop and replaced it with a custom, highly parallelized function. We used the Rayon data parallelism library to distribute the bitwise matrix multiplication across all available CPU cores.

By explicitly using a parallel mutable iterator and setting a minimum chunk size, we instructed the runtime to split the output matrix rows into independent tasks. These tasks are then processed concurrently by the global thread pool.

## Code Execution Map

Here is a diagram showing the difference between the old and new execution paths:

```mermaid
graph TD
    A[Inference Request] --> B{Backend Selected}
    B -->|Metal GPU| C[Fast GPU Path]
    B -->|CPU Fallback| D{Data Type}
    D -->|F16 / I8| E[Optimized Matmul]
    D -->|q1_0_g128| F[Old Execution Path]
    
    subgraph Old Path
    F --> F1[Sequential For Loop]
    F1 --> F2[Single Core Processing]
    F2 --> F3[High Latency]
    end
    
    D -->|q1_0_g128| G[New Execution Path]
    
    subgraph New Path
    G --> G1[Rayon Parallel Iterator]
    G1 --> G2[Core 1]
    G1 --> G3[Core 2]
    G1 --> G4[Core 3]
    G1 --> G5[Core N]
    G2 --> G6[Concurrent Bit Extraction]
    G3 --> G6
    G4 --> G6
    G5 --> G6
    G6 --> G7[Low Latency]
    end
```

## The Results

By parallelizing the execution, we achieved massive improvements:

* **Prefill time** dropped from 22.25 seconds to 2.13 seconds.
* **Decode latency** dropped from 3.14 seconds per token to 278 milliseconds per token.
* **CPU Usage** increased from 103 percent to nearly 600 percent, proving that the workload is now fully utilizing multiple cores.

This represents an 11x performance speedup, completely resolving the severe inference delays on the CPU backend.
