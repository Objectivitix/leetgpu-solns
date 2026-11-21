#include <cuda_runtime.h>

#include <algorithm>

#define LEAKY_COEFF 0.01f

__global__ void leaky_relu_kernel(const float *input, float *output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    float in = input[i];
    output[i] = (in > 0) ? in : in * LEAKY_COEFF;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaDeviceSynchronize();
}
