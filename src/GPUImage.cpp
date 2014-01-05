#include "GPUImage.h"
#include <stdio.h>
using namespace tcs_cuda;

GPUImage::GPUImage(int width, int height, rgbPixel* data) 
    : _width(width), _height(height), _deviceImagePtr(data) {}

GPUImage::~GPUImage() {

    //dealocate from gpu
    if (_deviceImagePtr != 0) {
        cudaFree(_deviceImagePtr);
        _deviceImagePtr = 0;
    }
}

int GPUImage::getWidth() const {
    return _width;
}

int GPUImage::getHeight() const {
    return _height;
}

rgbPixel* GPUImage::getDevicePixels() const {
    return _deviceImagePtr;
}

GPUImage GPUImage::load(const std::string& fileName) {

    //load image from file
    cv::Mat loadedImage = cv::imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);
    if (loadedImage.empty()) {
        std::cerr << "Couldn't open file: " << fileName << std::endl;
        exit(1);
    }

    if (!loadedImage.isContinuous()) {
        std::cerr << "Image isn't continuous! Exiting" << std::endl;
        exit(1);
    }

    rgbPixel* hostImagePtr = (rgbPixel*)loadedImage.ptr<unsigned char>(0);
    rgbPixel* deviceImagePtr = 0;

    const size_t allocSize = loadedImage.rows * loadedImage.cols * sizeof(rgbPixel);
    //alocate space on GPU required by image
    checkCudaErrors(cudaMalloc((void**)&deviceImagePtr, allocSize));
    //copy image data to GPU
    checkCudaErrors(cudaMemcpy(deviceImagePtr, hostImagePtr, allocSize, cudaMemcpyHostToDevice));
    
    //create a Image object
    return GPUImage(loadedImage.cols, loadedImage.rows, deviceImagePtr);
}

void GPUImage::save(const std::string& fileName, const GPUImage& image) {

    //load image from device to host
    const size_t allocSize = image._width * image._height * sizeof(rgbPixel);
    rgbPixel *hostImagePtr;
    checkCudaErrors(cudaMallocHost((void**)&hostImagePtr, allocSize));
    checkCudaErrors(cudaMemcpy(hostImagePtr, image._deviceImagePtr, allocSize, cudaMemcpyDeviceToHost));

    //create OpenCV image
    cv::Mat cvImage(image._height, image._width, CV_8UC3, (void*)hostImagePtr);

    //write image to file
    cv::imwrite(fileName.c_str(), cvImage);
}

GPUImage GPUImage::createEmpty(int width, int height) {
    if (width <= 0 || height <= 0) {
        std::cerr << "Non-positive image bounds!" << std::endl;
        exit(1);
    }

    rgbPixel* devicePtr = 0;
    const size_t allocSize = width * height * sizeof(rgbPixel);
    checkCudaErrors(cudaMalloc((void**)&devicePtr, allocSize));
    checkCudaErrors(cudaMemset(devicePtr, 0, allocSize));

    return GPUImage(width, height, devicePtr);
}
