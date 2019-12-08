---
title: "The CUDA Parallel Programming Model - 6. Memory Parallelism"
date: "2019-12-05T16:25:03.284Z"
---

DRAM bursting alone is not sufficient to realize the level of DRAM access bandwidth required by modern processors. In this post, I'll talk more about how to achieve better memory parallelism.

## Forms of Parallel Organization

- banks
- channels

![Channel and banks](./channel&banks.jpg)

- A processor contains one or more channels.
- Each channel is a **memory controller** with a **bus** that connects a set of **DRAM banks** to the processor.

### Bus

The data transfer bandwidth of a bus is defined by its _width_ and _clock frequency_.

Modern double data rate (DDR) busses perform two data transfers per clock cycle:

- one at the rising edge of each clock cycle
- one at the falling edge of each clock cycle

#### is DDR enough?

For example, a 64-bit DDR bus with a clock frequency of 1 GHz has a bandwidth of `8B*2*1 GHz =16 GB/sec`. This seems to be a large number but is often **too small** for modern CPUs and GPUs.

- A modern CPU might require a memory bandwidth of at least 32 GB/s, it's 2 channels for this example.
- a modern GPU might require 128 GB/s. For this example, it's 8 channels.

### Banks

The number of banks connected to a channel is determined by the what's required to **fully utilize the data transfer bandwidth of the bus**. This is illustrated in the picture below. Each bank contains an array of DRAM cells, the sensing amplifiers for accessing these cells, and the interface for delivering bursts of data to the bus.

![banks](./banks.jpg)

(More about interleaved data distribution later...)
