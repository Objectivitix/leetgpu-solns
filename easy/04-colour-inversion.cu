#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height) {
  int cell_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_index < width * height) {
    int red_index = cell_index * 4;
    int green_index = red_index + 1;
    int blue_index = green_index + 1;

    image[red_index] = 255 - image[red_index];
    image[green_index] = 255 - image[green_index];
    image[blue_index] = 255 - image[blue_index];
  }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char *image, int width, int height) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

  invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);

  cudaDeviceSynchronize();
}
