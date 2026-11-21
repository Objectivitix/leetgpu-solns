// https://leetgpu.com/challenges/relu

#include <cuda_runtime.h>

#include <algorithm>

__global__ void relu_kernel(const float *input, float *output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    output[i] = std::max(0.0f, input[i]);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaDeviceSynchronize();
}
