// https://leetgpu.com/challenges/reduction
//
// Parallel array reduction by iterative divide-and-conquer.
// Θ(lg N) time, assuming we have N/2 threads available.

#include <algorithm>
#include <cassert>

#include <cuda_runtime.h>

__global__ void reduce_step_kernel(float *in, float *out, int half_m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < half_m) {
    out[i] = in[2 * i] + in[2 * i + 1];
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  const auto input_size = N * sizeof(float);
  const int threads_per_block = 256;

  // Starting out as a copy of `input`, `arr` and `arr_out` will be
  // our "workspace" buffers, where the first `m` elements (see below)
  // of `arr` are intermediate values that remain to be reduced; we
  // use two buffers to avoid races in `reduce_step_kernel`.
  float *arr, *arr_out;

  assert(cudaMalloc((void **) &arr, input_size) == cudaSuccess);
  assert(cudaMalloc((void **) &arr_out, input_size) == cudaSuccess);

  cudaMemcpy(arr, input, input_size, cudaMemcpyDeviceToDevice);
  cudaMemcpy(arr_out, input, input_size, cudaMemcpyDeviceToDevice);

  // how many items there remain to be reduced;
  // when m == 1, arr[0] is our answer
  int m = N;

  while (m > 1) {
    const int half_m = m / 2;
    const int blocks_n = (half_m + threads_per_block - 1) / threads_per_block;

    reduce_step_kernel<<<blocks_n, threads_per_block>>>(arr, arr_out, half_m);

    if (m & 1) {
      // if m is odd, move the leftover to make pairs
      cudaMemcpy(
        arr_out + half_m, arr + half_m * 2,
        sizeof(float), cudaMemcpyDeviceToDevice
      );

      m = half_m + 1;
    } else {
      m = half_m;
    }

    // each reduce step depends on the previous,
    // so make sure all threads have finished
    cudaDeviceSynchronize();

    std::swap(arr, arr_out);
  }

  // write `arr[0]` to `output`
  cudaMemcpy(output, arr, sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(arr);
  cudaFree(arr_out);
}
