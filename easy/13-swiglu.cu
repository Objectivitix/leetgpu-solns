// how many LUs can there possibly be bro

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

__global__ void swiglu_kernel(const float *input, float *output, int halfN) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < halfN) {
    float in = input[i];
    output[i] = in * sigmoid(in) * input[halfN + i];
  }
}

// input, output are device pointers; N is guaranteed even
extern "C" void solve(const float *input, float *output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

  swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);

  cudaDeviceSynchronize();
}
