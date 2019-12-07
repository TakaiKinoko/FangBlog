---
title: "The CUDA Parallel Programming Model - 4.              
\n Syncthreads Examples"
date: "2019-12-07T13:25:03.284Z"
cover: "global.jpg"
---

This is the fourth post in a series about what I learnt in my GPU class at NYU this past fall. Here I collected several examples that showcase how the CUDA `__syncthreads()` command should (or should not) be used.

## Example 1

```c
__shared__ float partialSum[SIZE];
partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];
unsigned int t = threadIdx.x;
for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
     __syncthreads();
     if(t % (2*stride) == 0)
          partialSum[t] += partialSum[t+stride];
}
```

The `__syncthreads()` statement in the for-loop ensures that all partial sums for the previous iteration have been generated and before any one of the threads is allowed to begin the current iteration. This way, all threads that enter the second iteration will be using the values produced in the first iteration.