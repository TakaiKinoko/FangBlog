---
title: 'What I Learnt About The CUDA Parallel Programming Model'
date: '2019-12-06T22:12:03.284Z'
---


## Some Concepts

### three key abstractions

* a hierarchy of thread groups

* shared memories

* barrier synchronization

### granularity

* In parallel computing, granularity means the amount of __computation__ in relation to __communication (or transfer) of data__.

    * fine-grained: individual tasks are small in terms of code size and execution time.
    
    * coarse-grained: larger amounts of computation, infrequent data communication.

* CUDA abstraction:
    
    * __fine-grained__ data parallelism and thread parallelism nested within __roarse-grained__ data parallelism and task parallelism.

    * programmers partition the problem into __coarse sub-problems__ that can be solved independently in prallel by __blocks of threads__ and each sub-problem into finer pieces that can be solved cooperatively in parallel by __threads within the block__.

### kernel execution

* Executed in parallel by an array of threads, all of which run the same code.

* Each thread has an ID which is used to compute memory addresses and make control decisions.

### thread organization - grid of blocks

* Threads are arranged as a __grid__ of __blocks__.

* Blocks in a grid are completely __independent__ which means they can be executed in any order, in parallel or in series.

* The independence allows thread blocks to be scheduled in _any order_ across _any number of cores_.

### block

* Threads from the same block have access to a __shared memory__ .

* Execution of threads from the same block can be __synchronized__ (to coordinate memory accesses).

## CUDA Architecture

### SMs

The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors. 

Each SM has:

* a set of execution units
    
*  a set of registers 

* a chunch of shared memory.

### warp 

Warp is the __basic unit of execution__ in an NVIDIA GPU.

A warp is a collection of threads, 32 in current NVIDIA implementations. 

* threads within a warp a executed simultaneously by an SM. 

* multiple warps can be executed on an SM at once.

