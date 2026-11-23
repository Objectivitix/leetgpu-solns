// https://leetgpu.com/challenges/softmax
//
// A natural extension of the previous problem, sum reduction.
// Here we are asked to compute the softmax of a vector in
// Θ(lg N) time, assuming we have N/2 threads available. Since
// sum and max reduction both take Θ(lg N), and elementwise exp
// and division both take Θ(1), the solution follows naturally.

#include <algorithm>
#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

// Performs one step of an array-to-scalar reduction.
template <typename Func>
__global__ void reduce_step_kernel(
  const float *in, float *out, int half_m, Func reduce_op
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < half_m) {
    out[i] = reduce_op(in[2 * i], in[2 * i + 1]);
  }
}

// Reduces an array to a scalar in Θ(lg N) time using `reduce_op`.
//   pre: `reduce_op` is an associative binary operation.
//   device pointers: `input`, `output`.
//   effects: Writes to `output`.
template <typename Func>
void reduce(const float *input, float *output, int N, Func reduce_op) {
  const auto input_size = N * sizeof(float);
  const int threads_per_block = 256;

  float *arr, *arr_out;

  assert(cudaMalloc((void **) &arr, input_size) == cudaSuccess);
  assert(cudaMalloc((void **) &arr_out, input_size) == cudaSuccess);

  cudaMemcpy(arr, input, input_size, cudaMemcpyDeviceToDevice);

  // how many items there remain to be reduced;
  // when m == 1, arr[0] is our answer
  int m = N;

  while (m > 1) {
    const int half_m = m / 2;
    const int blocks_n = (half_m + threads_per_block - 1) / threads_per_block;

    reduce_step_kernel<<<blocks_n, threads_per_block>>>(
      arr, arr_out, half_m, reduce_op
    );

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

    cudaDeviceSynchronize();
    std::swap(arr, arr_out);
  }

  // write `arr[0]` to `output`
  cudaMemcpy(output, arr, sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(arr);
  cudaFree(arr_out);
}

// For each element `x` of `input`, stores fn(x)
// to the corresponding index in `output`.
template <typename Func>
__global__ void map_kernel(
  const float *input, float *output, int N, Func fn
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) output[i] = fn(input[i]);
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  float *max;
  assert(cudaMalloc((void **) &max, sizeof(float)) == cudaSuccess);

  reduce(input, max, N, [] __device__ (float x, float y) -> float {
    return std::max(x, y);
  });

  const int threads_per_block = 256;
  const int blocks_n = (N + threads_per_block - 1) / threads_per_block;

  map_kernel<<<blocks_n, threads_per_block>>>(
    input, output, N, [max] __device__ (float x) {
      return x - *max;
    }
  );

  cudaDeviceSynchronize();
  cudaFree(max);

  map_kernel<<<blocks_n, threads_per_block>>>(
    output, output, N, [] __device__ (float x) {
      return std::exp(x);
    }
  );

  cudaDeviceSynchronize();

  float *total;
  assert(cudaMalloc((void **) &total, sizeof(float)) == cudaSuccess);

  reduce(output, total, N, [] __device__ (float x, float y) -> float {
    return x + y;
  });

  map_kernel<<<blocks_n, threads_per_block>>>(
    output, output, N, [total] __device__ (float x) {
      return x / *total;
    }
  );

  cudaDeviceSynchronize();
  cudaFree(total);
}
