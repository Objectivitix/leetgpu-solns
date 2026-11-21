#include <cuda_runtime.h>

#include <cmath>

// Somewhat numerically stable implementation of the sigmoid
// function. No overflow here (but maybe underflow) :)
__device__ float sigmoid(float x) {
  if (x > 0) {
    float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = std::exp(x);
    return z / (1.0f + z);
  }
}

__global__ void silu_kernel(const float *input, float *output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    float in = input[i];
    output[i] = in * sigmoid(in);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaDeviceSynchronize();
}
