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
#define hdrPixel float
#define hdrChannelsNumber 3

namespace tcs_cuda {
    class GPUImage {
    private:
        int         _width;
        int         _height;
        rgbPixel*   _deviceImageRGBPtr;
		hdrPixel*   _deviceImageHDRPtr;

    private:
        GPUImage(int width, int height, rgbPixel* data);
        GPUImage(int width, int height, hdrPixel* data);

    public:
        ~GPUImage();

        int getWidth() const;
        int getHeight() const;

        rgbPixel* getDeviceRGBPixels() const;
        hdrPixel* getDeviceHDRPixels() const;

        static GPUImage loadRGB(const std::string& fileName);
        static GPUImage loadHDR(const std::string& fileName);
        
	static void saveRGB(const std::string& fileName, const GPUImage& image);
        static void saveHDR(const std::string& fileName, const GPUImage& image);

        static GPUImage createEmptyRGB(int width, int height);
        static GPUImage createEmptyHDR(int width, int height);
    };
}

#endif
