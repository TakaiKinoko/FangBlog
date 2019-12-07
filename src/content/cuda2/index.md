---
title: "What I Learnt About The CUDA Parallel Programming Model - 2"
date: "2019-12-06T23:12:03.284Z"
cover: "global.jpg"
---

This is the second post in a series about what I learnt in my GPU class at NYU this past fall. This will be mostly about **warps and SIMD hardward**.

## Kernel threads hierarchy

Recall that launching a CUDA kernel will generate a grid of threads organized as a **two-level** hierarchy.

1. top level: a 1/2/3-dimensional array of blocks.

1. bottom level: each block consists of a 1/2/3-dimensional array of threads.

## Synchronize threads?

Conceptually, threads in a block can execute in any order, just like blocks.

When an algorithm needs to execute in _phases_, **barrier synchronizations** should be used to ensure that all threads have completed a common phase before they start the next one.

But the correctness of executing a kernel should not depend on the synchrony amongst threads.

## Warp

Due to hardware cost considerations, CUDA devices currently bundle multiple threads for execution, which leads to performance limitations.
