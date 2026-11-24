// https://leetgpu.com/challenges/softmax-attention
// Put together everything you've done up to now!

#include <algorithm>
#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

// Performs one step of an array-to-scalar reduction for k
// arrays in parallel. `in` and `out` are both of shape k*N.
// `half_m` is floor(m / 2) where `m` is the number of items
// that we need to reduce from the start of each row.
template <typename Func>
__global__ void _batch_reduce_step_kernel(
  const float *in, float *out, int k, int N, int half_m,
  Func reduce_op
) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_offset = i * N;

  if (i < k && j < half_m) {
    out[row_offset + j] = reduce_op(
      in[row_offset + 2 * j], in[row_offset + 2 * j + 1]
    );
  }
}

__global__ void _batch_reduce_step_cleanup_kernel(
  const float *in, float *out, int k, int N, int half_m
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_offset = i * N;

  if (i < k) {
    out[row_offset + half_m] = in[row_offset + 2 * half_m];
  }
}

__global__ void _batch_reduce_write_output_kernel(
  const float *in, float *out, int k, int N
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_offset = i * N;

  if (i < k) {
    out[i] = in[row_offset + 0];
  }
}

// Reduces k arrays, each to a scalar, using `reduce_op`.
// `input` is of shape k*N, `output` is of shape k.
//   pre: `reduce_op` is an associative binary operation.
template <typename Func>
void batch_reduce(
  const float *input, float *output, int k, int N,
  Func reduce_op
) {
  const auto input_size = k * N * sizeof(float);

  // This split should adapt based on the relative sizes of
  // k and N; in this particular problem, when we call this
  // function, k and N are of the same order.
  const dim3 mat_threads_per_block(16, 16);

  const int threads_per_block = 256;
  const int blocks_n = (
    (k + threads_per_block - 1) / threads_per_block
  );

  float *arr, *arr_out;

  assert(cudaMalloc((void **) &arr, input_size) == cudaSuccess);
  assert(cudaMalloc((void **) &arr_out, input_size) == cudaSuccess);

  cudaMemcpy(arr, input, input_size, cudaMemcpyDeviceToDevice);

  // how many items there remain to be reduced;
  // when m == 1, arr[k * N + 0] are our answers
  int m = N;

  while (m > 1) {
    const int half_m = m / 2;
    const dim3 mat_blocks_n(
      (half_m + mat_threads_per_block.x - 1) / mat_threads_per_block.x,
      (k + mat_threads_per_block.y - 1) / mat_threads_per_block.y
    );

    _batch_reduce_step_kernel<<<mat_blocks_n, mat_threads_per_block>>>(
      arr, arr_out, k, N, half_m, reduce_op
    );

    cudaDeviceSynchronize();

    if (m & 1) {
      // if m is odd, move the leftover to make pairs
      _batch_reduce_step_cleanup_kernel<<<blocks_n, threads_per_block>>>(
        arr, arr_out, k, N, half_m
      );

      cudaDeviceSynchronize();

      m = half_m + 1;
    }

    else {
      m = half_m;
    }

    std::swap(arr, arr_out);
  }

  _batch_reduce_write_output_kernel<<<blocks_n, threads_per_block>>>(
    arr, output, k, N
  );

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

// For each element `x` of `input`, stores fn(x)
// to the corresponding index in `output`.
template <typename Func>
__global__ void row_map_kernel(
  const float *input, float *output, int k, int N, Func fn
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k * N) output[i] = fn(input[i], i / N);
}

// Computes softmax(input) row-wise and stores result in `output`.
void batch_softmax(const float *input, float *output, int k, int N) {
  float *max;
  assert(cudaMalloc((void **) &max, k * sizeof(float)) == cudaSuccess);

  batch_reduce(input, max, k, N, [] __device__ (float x, float y) -> float {
    return std::max(x, y);
  });

  const int threads_per_block = 256;
  const int blocks_n = (k * N + threads_per_block - 1) / threads_per_block;

  row_map_kernel<<<blocks_n, threads_per_block>>>(
    input, output, k, N, [max] __device__ (float x, int row) {
      return x - max[row];
    }
  );

  cudaDeviceSynchronize();
  cudaFree(max);

  map_kernel<<<blocks_n, threads_per_block>>>(
    output, output, k * N, [] __device__ (float x) {
      return std::exp(x);
    }
  );

  cudaDeviceSynchronize();

  float *total;
  assert(cudaMalloc((void **) &total, k * sizeof(float)) == cudaSuccess);

  batch_reduce(output, total, k, N, [] __device__ (float x, float y) -> float {
    return x + y;
  });

  row_map_kernel<<<blocks_n, threads_per_block>>>(
    output, output, k, N, [total] __device__ (float x, int row) {
      return x / total[row];
    }
  );

  cudaDeviceSynchronize();
  cudaFree(total);
}

// Computes A @ B (or A @ B^T if `Transpose`) and stores it in `C`.
// Shapes: A (M*K),  B (K*N, or N*K if `Transpose`), C (M*N).
// Too lazy to optimise this one.
template <bool Transpose = false>
__global__ void matmul_kernel(
  const float *A, const float *B, float *C,
  int M, int K, int N
) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < M && j < N) {
    float sum = 0;

    for (int k = 0; k < K; k++) {
      if constexpr (Transpose) {
        sum += A[i * K + k] * B[j * K + k];
      } else {
        sum += A[i * K + k] * B[k * N + j];
      }
    }

    C[i * N + j] = sum;
  }
}

// Computes Attention(Q, K, V) = softmax(QK^T/sqrt d)V, and stores
// it in `output`. Shapes: Q (M*d), K (N*d), V (N*d), output (M*d).
extern "C" void solve(
  const float* Q, const float* K, const float* V,
  float* output, int M, int N, int d
) {
  const float sqrt_d = std::sqrt(d);

  float *intermediate;  // MxN matrix holding many in-between results
  assert(
    cudaMalloc((void **) &intermediate, M * N * sizeof(float)) == cudaSuccess
  );

  const dim3 mat_threads_per_block(16, 16);
  const dim3 mat_blocks_n(
    (N + mat_threads_per_block.x - 1) / mat_threads_per_block.x,
    (M + mat_threads_per_block.y - 1) / mat_threads_per_block.y
  );

  matmul_kernel<true> <<<mat_blocks_n, mat_threads_per_block>>> (
    Q, K, intermediate, M, d, N
  );

  const int threads_per_block = 256;
  const int blocks_n = (M * N + threads_per_block - 1) / threads_per_block;

  map_kernel<<<blocks_n, threads_per_block>>>(
    intermediate, intermediate, M * N, [sqrt_d] __device__ (float x) {
      return x / sqrt_d;
    }
  );

  batch_softmax(intermediate, intermediate, M, N);

  matmul_kernel <<<mat_blocks_n, mat_threads_per_block>>> (
    intermediate, V, output, M, N, d
  );
}
