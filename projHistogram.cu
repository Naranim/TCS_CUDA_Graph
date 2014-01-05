#include "utils.h"
#include "GPUImage.h"
using namespace tcs_cuda;

__global__
void projCalculateHistogram_compute(
        rgbPixel* input, int size,
        unsigned int* histoR, unsigned int* histoG, 
        unsigned int* histoB, unsigned int* histoAll)
{
    __shared__ unsigned int Red[256];
    __shared__ unsigned int Green[256];
    __shared__ unsigned int Blue[256];

    int offset = gridDim.x * blockDim.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < size) {
        atomicAdd( &Red[input[i].z], 1 );
        atomicAdd( &Green[input[i].y], 1 );
        atomicAdd( &Blue[input[i].x], 1 );
        i += offset;
    }

    __syncthreads();

    atomicAdd( &histoR[threadIdx.x], Red[threadIdx.x] );
    atomicAdd( &histoG[threadIdx.x], Green[threadIdx.x] );
    atomicAdd( &histoB[threadIdx.x], Blue[threadIdx.x] );
    
    atomicAdd( &histoAll[threadIdx.x], Red[threadIdx.x] );
    atomicAdd( &histoAll[threadIdx.x], Green[threadIdx.x] );
    atomicAdd( &histoAll[threadIdx.x], Blue[threadIdx.x] );
}

__global__
void projCalculateHistogram_generateImage(
        unsigned int* histogram, rgbPixel* output, int width, int height) 
{
    int thId = blockDim.x * blockIdx.x + threadIdx.x;
    if (thId >= width * height)
        return;

    unsigned int maxVal = 0;
    for (int i=0;i<256;++i)
        if (histogram[i] > maxVal) 
            maxVal = histogram[i];

    int tX = thId % width;
    int tY = thId / width;

    int valX = (256 * tX) / width; 
    if (valX > 255) valX = 255;
    if (valX < 0) valX = 0;

    int maxH = (histogram[tX] * height) / maxVal;

    if (height - maxH > tY) {
        output[thId].x = 255;
        output[thId].y = 255;
        output[thId].z = 255;
    } else {
        output[thId].x = 15;
        output[thId].y = 65;
        output[thId].z = 175;
    }
}

GPUImage histogramTransform(const GPUImage& input) {
    return GPUImage::createEmpty(512, 256);
}

void projHistogram(const GPUImage& input, GPUImage& output, int option) {
    rgbPixel* dInput = input.getDevicePixels();
    rgbPixel* dOutput = output.getDevicePixels();
    

    int histoSize = sizeof(unsigned int) * 256;
    unsigned int *histoR, *histoG, *histoB, *histoAll;

    checkCudaErrors(cudaMalloc((void**)&histoR, histoSize));
    checkCudaErrors(cudaMalloc((void**)&histoG, histoSize));
    checkCudaErrors(cudaMalloc((void**)&histoB, histoSize));
    checkCudaErrors(cudaMalloc((void**)&histoAll, histoSize));
    checkCudaErrors(cudaMemset(histoR, 0, histoSize));
    checkCudaErrors(cudaMemset(histoG, 0, histoSize));
    checkCudaErrors(cudaMemset(histoB, 0, histoSize));
    checkCudaErrors(cudaMemset(histoAll, 0, histoSize));

    int size = input.getWidth() * input.getHeight();
    
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((size + 255) / 256, 1, 1);
    projCalculateHistogram_compute<<<gridSize, blockSize>>>(dInput, size, histoR, histoG, histoB, histoAll);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    int width = output.getWidth();
    int height = output.getHeight();
    size = width * height;
    
    const dim3 blSize(1024, 1, 1);
    const dim3 grSize((size + 1023) / 1024, 1, 1);

    unsigned int* histo = histoAll;
    switch(option) {
        case 1: histo = histoR; break;
        case 2: histo = histoG; break;
        case 3: histo = histoB; break;
        default: histo = histoAll; break;
    }
    projCalculateHistogram_generateImage<<<grSize, blSize>>>(histo, dOutput, width, height);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaFree(histoR));
    checkCudaErrors(cudaFree(histoG));
    checkCudaErrors(cudaFree(histoB));
    checkCudaErrors(cudaFree(histoAll));
}

void projHistogram_All(const GPUImage& input, GPUImage& output) {
    projHistogram(input, output, 4);
}

void projHistogram_Red(const GPUImage& input, GPUImage& output) {
    projHistogram(input, output, 1);
}

void projHistogram_Green(const GPUImage& input, GPUImage& output) {
    projHistogram(input, output, 2);
}

void projHistogram_Blue(const GPUImage& input, GPUImage& output) {
    projHistogram(input, output, 3);
}
