---
title: "Recap: GPU Latency Tolerance and Zero-Overhead Thread-Scheduling"
date: "2019-12-10T10:10:03.284Z"
---

I briefly talked about how CUDA processors hide long-latency operations such as global memory accesses through their warp-scheduling mechanism in [The CUDA Parallel Programming Model - 2. Warps](/cuda2-warp).

## Recap

Unlike CPUs, GPUs do not dedicate as much chip area to cache memories and branch prediction mechanisms. This is because GPUs have the ability to tolerate long-latency operations.

GPU SMs are designed in the way that each SM can execute only a small number of warps at any given time. However, the number of warps residing on the SM is much bigger than what can actually be executed. The reason for this is, when a warp is currently waiting for result from a **long-latency operation**, such as:

- global memory access
- floating-point arithmetic
- branch instructions

the warp scheduler on the SM will pick another warp that's ready to execute, therefore avoids idle time. By having a sufficient number of warps on the SM, the hardware will likely fo find a warp to execute at any point in time.

Having zero idle time or wasted time is referred to as **zero-overhead thread scheduling** in processor designs.

## Question

Assume that a CUDA device allows up to 8 blocks and 1024 threads per SM, whichever becomes a limitation first. Furthermore, it allows up to 512 threads in each block. Should we use 8×8, 16×16, or 32×32 thread blocks?

To answer the question, we can analyze the pros and cons of each choice.

If we use 8 ×8 blocks, each block would have only 64 threads. We will need 1024/64 =12 blocks to fully occupy an SM. However, each SM can only allow up to 8 blocks; thus, we will end up with only 64 ×8 =512 threads in each SM. This limited number implies that the SM execution resources will likely be underutilized because fewer warps will be available to schedule around long-latency operations.

The 16 ×16 blocks result in 256 threads per block, implying that each SM can take 1024/256 =4 blocks. This number is within the 8-block limitation and is a good configuration as it will allow us a **full thread capacity in each SM** and a **maximal number of warps** for scheduling around the long-latency operations.

The 32 ×32 blocks would give 1024 threads in each block, which exceeds the 512 threads per block limitation of this device. Only 16 ×16 blocks allow a maximal number of threads assigned to each SM.
