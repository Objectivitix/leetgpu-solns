// https://leetgpu.com/challenges/reduction
//
// Parallel array reduction by iterative divide-and-conquer.
// Θ(lg N) time, assuming we have N/2 threads available.

#include <cassert>

#include <cuda_runtime.h>

__global__ void reduce_step_kernel(float *arr, int half_m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < half_m) {
    arr[i] = arr[2 * i] + arr[2 * i + 1];
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  const auto input_size = N * sizeof(float);
  const int threads_per_block = 256;

  // starting out as a copy of `input`, `arr` will be our
  // "workspace" where the first `m` elements (see below)
  // are intermediate values that remain to be reduced
  float *arr;

  if (cudaMalloc((void **) &arr, input_size) != cudaSuccess) {
    assert(false);
  }

  cudaMemcpy(arr, input, input_size, cudaMemcpyDeviceToDevice);

  // how many items there remain to be reduced;
  // when m == 1, arr[0] is our answer
  int m = N;

  while (m > 1) {
    const int half_m = m / 2;
    const int blocks_n = (half_m + threads_per_block - 1) / threads_per_block;

    reduce_step_kernel<<<blocks_n, threads_per_block>>>(arr, half_m);
    cudaDeviceSynchronize();

    if (m & 1) {
      // if m is odd, move the leftover to make pairs
      cudaMemcpy(
        arr + half_m, arr + half_m * 2,
        sizeof(float), cudaMemcpyDeviceToDevice
      );

      m = half_m + 1;
      continue;
    }

    m = half_m;
  }

  // write `arr[0]` to `output`
  cudaMemcpy(output, arr, sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(arr);
}
