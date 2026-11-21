// https://leetgpu.com/challenges/rgb-to-grayscale

#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(
  const float *input, float *output,
  int width, int height
) {
  int cell_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_index < width * height) {
    int red_index = cell_index * 3;
    int green_index = red_index + 1;
    int blue_index = green_index + 1;

    output[cell_index] = (
      0.299 * input[red_index]
      + 0.587 * input[green_index]
      + 0.114 * input[blue_index]
    );
  }
}

// input, output are device pointers
extern "C" void solve(
  const float *input, float *output,
  int width, int height
) {
  int total_pixels = width * height;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

  rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    input, output, width, height
  );

  cudaDeviceSynchronize();
}
