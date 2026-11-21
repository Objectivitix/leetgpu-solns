// https://leetgpu.com/challenges/reverse-array

#include <cuda_runtime.h>

__constant__ int _half_n;

__global__ void reverse_array(float *input, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < _half_n) {
    float temp = input[i];
    input[i] = input[N - 1 - i];
    input[N - 1 - i] = temp;
  }
}

// input is device pointer
extern "C" void solve(float *input, int N) {
  int half_n = N / 2;

  int threadsPerBlock = 256;
  int blocksPerGrid = (half_n + threadsPerBlock - 1) / threadsPerBlock;

  cudaMemcpyToSymbol(_half_n, &half_n, sizeof(int));

  reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);

  cudaDeviceSynchronize();
}
