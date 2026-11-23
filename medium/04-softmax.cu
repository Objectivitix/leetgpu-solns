// https://leetgpu.com/challenges/softmax
//
// A natural extension of the previous problem, sum reduction.
// Here we are asked to compute the softmax of a vector in
// Θ(lg N) time, assuming we have N/2 threads available. Since
// sum and max reduction both take Θ(lg N), and elementwise exp
// and division both take Θ(1), the solution follows naturally.

#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

// Performs one step of an array-to-scalar reduction.
template <typename Func>
__global__ void reduce_step_kernel(
  float *in, float *out, int half_m, Func &&reduce_op
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
void reduce(const float *input, float *output, int N, Func &&reduce_op) {
  const auto input_size = N * sizeof(float);
  const int threads_per_block = 256;

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

// Subtracts `c` from each element of `input`,
// and stores the result in `output`.
__global__ void subtract_kernel(
  const float *input, float *output, int N, float *c
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) output[i] = input[i] - *c;
}

// Exponentiates each element of `arr`.
__global__ void exp_kernel(float *arr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) arr[i] = std::exp(arr[i]);
}

// Divides each element of `arr` by `c`.
__global__ void divide_kernel(float *arr, int N, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) arr[i] /= *c;
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

  subtract_kernel<<<blocks_n, threads_per_block>>>(
    input, output, N, max
  );

  cudaDeviceSynchronize();
  cudaFree(max);

  exp_kernel<<<blocks_n, threads_per_block>>>(output, N);
  cudaDeviceSynchronize();

  float *total;
  assert(cudaMalloc((void **) &total, sizeof(float)) == cudaSuccess);

  reduce(output, total, N, [] __device__ (float x, float y) -> float {
    return x + y;
  });

  divide_kernel<<<blocks_n, threads_per_block>>>(output, N, total);
  cudaDeviceSynchronize();

  cudaFree(total);
}
