---
title: "Some CUDA Related Q&As"
date: "2019-12-08T10:10:03.284Z"
---

## Review

### CUDA's Physical Architecture

#### SMs

CUDA-capable GPU cards are composed of one or more **Streaming Multiprocessors (SMs)**, which are an _abstraction_ of the underlying hardware.

#### cores

Each SM has a set of **Streaming Processors (SPs)**, also called CUDA cores, which **share a cache of shared memory** that is faster than the GPU’s global memory but that can only be accessed by the threads running on the SPs the that SM. These streaming processors are the “cores” that execute instructions.

#### cores/SM

The numbers of SPs/cores in an SM and the number of SMs depend on your device.

It is important to realize, however, that regardless of GPU model, there **are many more CUDA cores in a GPU than in a typical multicore CPU**: hundreds or thousands more. For example, the Kepler Streaming Multiprocessor design, dubbed SMX, contains 192 single-precision CUDA cores, 64 double-precision units, 32 special function units, and 32 load/store units.

#### warps, SM and cores

CUDA **cores are grouped together** to perform instructions in a **warp** of threads. Warp simply means a group of threads that are scheduled together to execute the same instructions in lockstep.

Depending on the model of GPU, the **cores** may be double or quadruple pumped so that they **execute one instruction on two or four threads** in as many clock cycles.

_For instance, Tesla devices use a group of 8 quadpumped cores to execute a single warp. If there are less than 32 threads scheduled in the warp, it will still take as long to execute the instructions._

All CUDA cards to date use a warp size of 32.

Each **SM** has at least one **warp scheduler**, which is responsible for executing 32 threads.

#### tiling

The programmer is responsible for ensuring that the threads are being assigned efficiently for code that is designed to run on the GPU. The assignment of threads is done virtually in the code using what is sometimes referred to as a ‘tiling’ scheme of blocks of threads that form a grid. Programmers define a kernel function that will be executed on the CUDA card using a particular tiling scheme.

### CUDA's Virtual Architecture

When programming in CUDA we work with blocks of threads and grids of blocks. What is the relationship between this virtual architecture and the CUDA card’s physical architecture?

#### block

When kernels are launched, each **block** in a grid is assigned to a **Streaming Multiprocessor**. This allows threads in a block to use **shared** memory. If a block doesn’t use the full resources of the SM then multiple blocks may be assigned at once. If all of the SMs are busy then the extra blocks will have to wait until a SM becomes free.

#### threads

_Once a block is assigned to an SM_, it’s threads are split into warps by the warp scheduler and executed on the CUDA cores.

Since the same instructions are executed on each thread in the warp simultaneously it’s generally a bad idea to have conditionals in kernel code. This type of code is sometimes called **divergent**: when some threads in a warp are unable to execute the same instruction as other threads in a warp, those threads are diverged and do no work.

#### warp's context switch

Due to the hugely increased number of registers, a **warp’s context (it’s registers, program counter etc.) stays on chip for the life of the warp**. This means there is no additional cost to switching between warps vs executing the next step of a given warp. This allows the GPU to switch to hide some of it’s memory latency by switching to a new warp while it waits for a costly read.

## Block Assignment

1. Before a block is assgined to an SM, it's given all the resources it needs beforehands.

   These resources include:

   - shared memory

   - registers

   - a slot in the SM scheduler

   ![block](./block.png)

1. Why not give blocks SPs too?

   - There are way more blocks than SPs!

1. Why don't we assign these resources after the block is assigned?

   First, more on CUDA runtime:

   ### Notes On CUDA Runtime

   - The runtime system:

     - maintains a list of blocks to be executed

     - assigns new blocks to SM as they compute previously assigned blocks

   - CUDA runtime automatically reduces the number of blocks assgined to each SM until resource usage is under limit.

   Then, coming back to the point that we have **zero-cost context switch for warps** now, preassign resources to blocks (which in terms of execution is a bunch of warps) will make it that when warps are scheduled, their resources are already on-chip. So effectively we can achieve **zero-cost scheduling**.

