#ifndef __TCS_CUDA_PROJ_GPU_IMAGE_H_INCLUDED_
#define __TCS_CUDA_PROJ_GPU_IMAGE_H_INCLUDED_

#pragma once

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

//STD
#include <string>
#include <iostream>

#define grayPixel uchar
#define rgbPixel uchar3
#define rgbaPixel uchar4

namespace tcs_cuda {
    class GPUImage {
    private:
        int         _width;
        int         _height;
        rgbPixel*   _deviceImagePtr;

    private:
        GPUImage(int width, int height, rgbPixel* data);

    public:
        ~GPUImage();

        int getWidth() const;
        int getHeight() const;

        rgbPixel* getDevicePixels() const;

        static GPUImage load(const std::string& fileName);
        static void save(const std::string& fileName, const GPUImage& image);

        static GPUImage createEmpty(int width, int height);
    };
}

#endif
