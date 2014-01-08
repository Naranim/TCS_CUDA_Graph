#include "utils.h"
#include "projTransform.h"
#include "GPUImage.h"

#include <algorithm>

#define PIXELS_PER_TH 128

using namespace tcs_cuda;

enum direction {
    HORIZONTAL,
    VERTICAL,
    LEFT,
    RIGHT
};

__global__
void projFlip(rgbPixel* input,
              rgbPixel* output,
              int width,
              int height,
              direction d) {

    const int thId = blockDim.x * blockIdx.x + threadIdx.x;
    const int startInd = PIXELS_PER_TH * thId;
    const int endInd = (startInd + PIXELS_PER_TH < width * height) ?
                        startInd + PIXELS_PER_TH : width * height;

    for(int i = startInd; i < endInd; ++i) {
        int x = i / width;
        int y = i % width;

        if(d == HORIZONTAL) {
            // output[x][width - y - 1] = input[x][y];
            output[x*width + (width - y - 1)] = input[x*width + y];
        }
        if(d == VERTICAL) {
            // output[height - x - 1][y] = input[x][y];
            output[(height - x - 1)*width + y] = input[x*width + y];
        }
    }
}

void flip(const GPUImage& input, GPUImage& output, direction d) {
    rgbPixel* cuInput = input.getDeviceRGBPixels();
    rgbPixel* cuOutput = output.getDeviceRGBPixels();
    int width = input.getWidth();
    int height = input.getHeight();

    const dim3 blockSize(1024, 1, 1);
    const int gridXSize = (width * height + PIXELS_PER_TH + 1024 - 1) / (PIXELS_PER_TH + 1024);
    const dim3 gridSize(gridXSize , 1, 1);
    projFlip<<<gridSize, blockSize>>>(cuInput, cuOutput, width, height, d);
  
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

void flipVer(const GPUImage& input, GPUImage& output) {
    flip(input, output, VERTICAL);
}

void flipHor(const GPUImage& input, GPUImage& output) {
    flip(input, output, HORIZONTAL);
}

__global__
void projRotate(rgbPixel* input,
              rgbPixel* output,
              int width,
              int height,
              direction d) {

    const int thId = blockDim.x * blockIdx.x + threadIdx.x;
    const int startInd = PIXELS_PER_TH * thId;
    const int endInd = (startInd + PIXELS_PER_TH < width * height) ?
                        startInd + PIXELS_PER_TH : width * height;

    for(int i = startInd; i < endInd; ++i) {
        int x = i / width;
        int y = i % width;

        if(d == LEFT) {
            // output[width - y - 1][x] = input[x][y];
            output[(width - y - 1)*height + x] = input[x*width + y];   
        }
        if(d == RIGHT) {
            // output[y][height - x - 1] = input[x][y];
            output[y*height + (height - x - 1)] = input[x*width + y];   
        }
    }
}

GPUImage rotateTransform(const GPUImage& input) {
    return GPUImage::createEmptyRGB(input.getHeight(), input.getWidth());
}

void rotate(const GPUImage& input, GPUImage& output, direction d) {
    rgbPixel* cuInput = input.getDeviceRGBPixels();
    rgbPixel* cuOutput = output.getDeviceRGBPixels();
    int width = input.getWidth();
    int height = input.getHeight();

    const dim3 blockSize(1024, 1, 1);
    const int gridXSize = (width * height + PIXELS_PER_TH + 1024 - 1) / (PIXELS_PER_TH + 1024);
    const dim3 gridSize(gridXSize , 1, 1);
    projRotate<<<gridSize, blockSize>>>(cuInput, cuOutput, width, height, d);
  
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

void rotateLeft(const GPUImage& input, GPUImage& output) {
    rotate(input, output, LEFT);
}

void rotateRight(const GPUImage& input, GPUImage& output) {
    rotate(input, output, RIGHT);
}
