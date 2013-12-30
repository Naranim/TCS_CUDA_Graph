#include "utils.h"
#include "GPUImage.h"
using namespace tcs_cuda;

__global__
void projCalculateInvert(rgbPixel* input, rgbPixel* output, int size)
{
    int thId = blockDim.x * blockIdx.x + threadIdx.x;

    if (thId >= size)
        return;

    unsigned char R = input[thId].x;
    unsigned char G = input[thId].y;
    unsigned char B = input[thId].z;

    output[thId].x = 255 - R;
    output[thId].y = 255 - G;
    output[thId].z = 255 - B;
}

void projInvert(const GPUImage& input, GPUImage& output) {
    rgbPixel* dInput = input.getDevicePixels();
    rgbPixel* dOutput = output.getDevicePixels();
    int size = input.getWidth() * input.getHeight();

    const dim3 blockSize(1024, 1, 1);
    const dim3 gridSize((size + 1023) / 1024, 1, 1);
    projCalculateInvert<<<gridSize, blockSize>>>(dInput, dOutput, size);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

