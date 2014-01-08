#include "utils.h"
#include "GPUImage.h"
using namespace tcs_cuda;

__global__
void projCalculateMatrix3x3(
        rgbPixel* input, rgbPixel* output, 
        int width, int height, int* matrix3x3)
{
    //compute coordinates
    int bx = threadIdx.x, by = threadIdx.y;
    int tx = blockIdx.x * 30 + bx - 1;
    int ty = blockIdx.y * 30 + by - 1;
    int cell = ty * width + tx;

    bool isOutside = (tx < 0 || tx >= width || ty < 0 || ty >= height);

    //fill in shared
    __shared__ rgbPixel sh[32][32];
    if (isOutside) sh[bx][by].x = sh[bx][by].y = sh[bx][by].z = 0;
    else sh[bx][by] = input[cell];
    
    //check if (tx,ty) is inside image if not skip computation
    if (isOutside)
        return;

    //sumCoefs
    int sumCoefs = 0;
    for (int i=0;i<9;++i)
        sumCoefs += matrix3x3[i];
    
    __syncthreads();

    //this thread is shared border - skip
    if (bx == 0 || bx == 31 || by == 0 || by == 31)
        return;

    //sumVals
    int sumVals[3] = { 0, 0, 0 };
    for (int dy=by-1; dy<=by+1; ++dy) {
        for (int dx=bx-1; dx<=bx+1; ++dx) {
            int coef = matrix3x3[3 * (dy-by+1) + (dx-bx+1)];
            sumVals[0] += sh[dx][dy].x * coef;
            sumVals[1] += sh[dx][dy].y * coef;
            sumVals[2] += sh[dx][dy].z * coef;
        }
    }

    __syncthreads();

    //result
    int result[3] = { sumVals[0], sumVals[1], sumVals[2] };
    if (sumCoefs != 0) {
        for (int i=0;i<3;++i)
            result[i] = sumVals[i] / sumCoefs;
    }

    rgbPixel val;
    val.x = min(255, max(0, result[0]));
    val.y = min(255, max(0, result[1]));
    val.z = min(255, max(0, result[2]));

    output[cell] = val;
}

void projMatrix3x3(const GPUImage& input, GPUImage& output, int* matrix3x3) {
    rgbPixel* dInput = input.getDeviceRGBPixels();
    rgbPixel* dOutput = output.getDeviceRGBPixels();
    int width = input.getWidth();
    int height = input.getHeight();

    int* gpuCoefs;
    checkCudaErrors(cudaMalloc((void**)&gpuCoefs, sizeof(int)*9));
    checkCudaErrors(cudaMemcpy(gpuCoefs, matrix3x3, sizeof(int)*9, cudaMemcpyHostToDevice));

    const dim3 blockSize(32, 32, 1);
    const dim3 gridSize((width + 29) / 30, (height + 29) / 30, 1);
    projCalculateMatrix3x3<<<gridSize, blockSize>>>(dInput, dOutput, width, height, gpuCoefs);
    
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(gpuCoefs));
}

