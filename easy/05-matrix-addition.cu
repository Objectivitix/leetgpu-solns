// https://leetgpu.com/challenges/matrix-addition

#include <cuda_runtime.h>

__constant__ int n_squared;

__global__ void matrix_add(const float *A, const float *B, float *C) {
  int flat_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (flat_index < n_squared) {
    C[flat_index] = A[flat_index] + B[flat_index];
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;
  int n_sq = N * N;

  int blocksPerGrid = (n_sq + threadsPerBlock - 1) / threadsPerBlock;

  cudaMemcpyToSymbol(n_squared, &n_sq, sizeof(int));

  matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);

  cudaDeviceSynchronize();
}
