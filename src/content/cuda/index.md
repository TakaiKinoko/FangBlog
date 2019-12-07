---
title: "What I Learnt About The CUDA Parallel Programming Model"
date: "2019-12-06T22:12:03.284Z"
---

## TABLE OF CONTENTS

1. Concepts
   1. key abstractions
   1. granularity
1. CUDA Architecture
   1. kernel execution
   1. thread organization
   1. blocks
   1. SMs
   1. warp
   1. execution picture
1. Thread ID
1. Memory Hierarchy

## Some Concepts

### three key abstractions

- a hierarchy of thread groups

- shared memories

- barrier synchronization

### granularity

- In parallel computing, granularity means the amount of **computation** in relation to **communication (or transfer) of data**.

  - fine-grained: individual tasks are small in terms of code size and execution time.

  - coarse-grained: larger amounts of computation, infrequent data communication.

- CUDA abstraction:

  - **fine-grained** data parallelism and thread parallelism nested within **roarse-grained** data parallelism and task parallelism.

  - programmers partition the problem into **coarse sub-problems** that can be solved independently in prallel by **blocks of threads** and each sub-problem into finer pieces that can be solved cooperatively in parallel by **threads within the block**.

## CUDA Architecture

### kernel execution

- Executed in parallel by an array of threads, all of which run the same code.

- Each thread has an ID which is used to compute memory addresses and make control decisions.

### thread organization - grid of blocks

- Threads are arranged as a **grid** of **blocks**.

- Blocks in a grid are completely **independent** which means they can be executed in any order, in parallel or in series.

- The independence allows thread blocks to be scheduled in _any order_ across _any number of cores_.

### block

- Threads from the same block have access to a **shared memory** .

- Execution of threads from the same block can be **synchronized** (to coordinate memory accesses).

### SMs

The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors.

Each SM has:

- a set of execution units
- a set of registers

- a chunch of shared memory.

### 🧐warp

Warp is the **basic unit of execution** in an NVIDIA GPU.

A warp is a collection of threads, 32 in current NVIDIA implementations.

- threads within a warp a executed simultaneously by an SM.

- multiple warps can be executed on an SM at once.

The mapping between warps and thread blocks can affect the performance of the kernel.
**It's usually good the keep the size of a block a multiple of 32**.

### picture the process of execution

1. CUDA program on the host CPU invokes a **kernel grid**

1. blocks in the grid are enumerated and distributed to SMs with available execution capacity

1. the threads of a block execute concurrently on one SM

1. as thread blocks terminate, new blocks are launched on the vacated SMs

## Thread ID

TODO

## Memory Hierarchy

### between CPU and GPU

CPU and GPU has **separate memory spaces** => data must be moved from CPU to GPU before computation starts, as well as moved back to CPU once processing has completed.

### global memory

- accessible to all threads as well as the host (CPU)
