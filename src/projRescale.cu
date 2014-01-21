#include "projRescale.h"

#include <cstdio>

#define IND(a,b,n) n*a+b

__device__
rgbPixel mul(rgbPixel p, double d) {
    p.x *= d;
    p.y *= d;
    p.z *= d;
    return p;
}

__device__
rgbPixel add(rgbPixel p, rgbPixel q) {
    p.x += q.x;
    p.y += q.y;
    p.z += q.z;
    return p;
}

__device__
rgbPixel calc(rgbPixel* input, int n, int x, int y, double a, double b) {
    rgbPixel tmp = input[IND(x,y,n)];
    return mul(tmp, a*b);  
}

__global__
void projRotate(rgbPixel* input,
                rgbPixel* output,
                int inWidth,
                int inHeight,
                int outWidth,
                int outHeight,
                double scale) {

    int N = outWidth*outHeight;
    int index = (blockIdx.x * blockDim.x + threadIdx.x)*32;
    for(int i = 0; index < N && i < 32; index++, i++)  {
        int x = index / outWidth;
        int y = index % outWidth;

        double inX = x / scale;
        double inY = y / scale;

        int nearestX = static_cast<int>(inX + 0.5);
        int nearestY = static_cast<int>(inY + 0.5);

        double distX = inX - nearestX + 0.5;
        double distY = inY - nearestY + 0.5;

        rgbPixel ret;

        ret.x = 0;
        ret.y = 0;
        ret.z = 0;

        ret = add(ret, calc(input, inWidth, nearestX, nearestY, 1-distX, 1-distY));
        ret = add(ret, calc(input, inWidth, nearestX+1, nearestY, distX, 1-distY));
        ret = add(ret, calc(input, inWidth, nearestX, nearestY+1, 1-distX, distY));
        ret = add(ret, calc(input, inWidth, nearestX+1, nearestY+1, distX, distY));

        output[index] = ret;
    }
}

void rescale(const GPUImage& input, GPUImage& output) {
    double scaleX, scaleY, scale;
    scaleX = ((double)output.getWidth())/((double)input.getWidth());
    scaleY = ((double)output.getHeight())/((double)input.getHeight());
    scale = (scaleX + scaleY) / 2;

    rgbPixel* cuInput = input.getDeviceRGBPixels();
    rgbPixel* cuOutput = output.getDeviceRGBPixels();

    int inWidth = input.getWidth();
    int inHeight = input.getHeight();
    int outWidth = output.getWidth();
    int outHeight = output.getHeight();

    dim3 gridSize((outHeight*outWidth+32767) >> 15);
    dim3 blockSize(1024);
    projRotate<<<gridSize, blockSize>>>(cuInput, cuOutput, inWidth, inHeight,
                                        outWidth, outHeight, scale);

}

GPUImage rescaleTransform(const GPUImage& input) {
    double scale;
    printf("\nSet the scale: ");
    scanf("%lf", &scale);
    double W = scale * (double)(input.getWidth());
    double H = scale * (double)(input.getHeight());
    return GPUImage::createEmptyRGB(W, H);
}
