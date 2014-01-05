#include "utils.h"
#include "GPUImage.h"
using namespace tcs_cuda;

__global__
void projCalculateGreyScale(
        rgbPixel* rgbaImage,
        rgbPixel* greyImage,
        int numRows, int numCols)
{
  int thidId = blockIdx.x * blockDim.x + threadIdx.x;  
  if (thidId >= numRows * numCols)
      return;
    
  unsigned char R = rgbaImage[thidId].x;
  unsigned char G = rgbaImage[thidId].y;
  unsigned char B = rgbaImage[thidId].z;
  greyImage[thidId].x = greyImage[thidId].y = greyImage[thidId].z = .299f * R + .587f * G + .114f * B;  
}

void projGreyscale(const GPUImage& input, GPUImage& output)
{
    rgbPixel* d_rgbaImage = input.getDevicePixels();
    rgbPixel* d_greyImage = output.getDevicePixels();
    int numRows = input.getHeight();
    int numCols = input.getWidth();

    const dim3 blockSize(1024, 1, 1);
    const dim3 gridSize(numRows * numCols / 1024.0 + 1, 1, 1);
    projCalculateGreyScale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}
