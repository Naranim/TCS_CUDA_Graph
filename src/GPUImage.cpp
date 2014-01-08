#include "GPUImage.h"
#include <stdio.h>
using namespace tcs_cuda;

GPUImage::GPUImage(int width, int height, rgbPixel* data) 
    : _width(width), _height(height), _deviceImageRGBPtr(data), _deviceImageHDRPtr(0) {}

GPUImage::GPUImage(int width, int height, hdrPixel* data) 
    : _width(width), _height(height), _deviceImageHDRPtr(data), _deviceImageRGBPtr(0) {}

GPUImage::~GPUImage() {

    //dealocate from gpu
    if (_deviceImageRGBPtr != 0) {
        cudaFree(_deviceImageRGBPtr);
        _deviceImageRGBPtr = 0;
    }

    //dealocate from gpu
    if (_deviceImageHDRPtr != 0) {
        cudaFree(_deviceImageHDRPtr);
        _deviceImageHDRPtr = 0;
    }
}

int GPUImage::getWidth() const {
    return _width;
}

int GPUImage::getHeight() const {
    return _height;
}

rgbPixel* GPUImage::getDeviceRGBPixels() const {
    return _deviceImageRGBPtr;
}

hdrPixel* GPUImage::getDeviceHDRPixels() const {
    return _deviceImageHDRPtr;
}

GPUImage GPUImage::loadRGB(const std::string& fileName) {

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


GPUImage GPUImage::loadHDR(const std::string& fileName)
{
	cv::Mat loadedImage = cv::imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
	if (loadedImage.empty()) {
		std::cerr << "Couldn't open file: " << fileName << std::endl;
		exit(1);
	}

	if (loadedImage.channels() != hdrChannelsNumber) {
		std::cerr << "Image must be color!" << std::endl;
		exit(1);
	}

	if (!loadedImage.isContinuous()) {
		std::cerr << "Image isn't continuous!" << std::endl;
		exit(1);
	}

	size_t pixelsNumber = loadedImage.rows * loadedImage.cols * loadedImage.channels();
    hdrPixel* hostImagePtr = new hdrPixel[pixelsNumber];
	unsigned char* loadedDataPtr = (unsigned char*)loadedImage.ptr<unsigned char>(0);
	for (size_t i = 0; i < size_t(pixelsNumber); i++){
		hostImagePtr[i] = ((hdrPixel)loadedDataPtr[i]);	
	}
	
    hdrPixel* deviceImagePtr = 0;

    const size_t allocSize = pixelsNumber * sizeof(hdrPixel);
    //alocate space on GPU required by image
    checkCudaErrors(cudaMalloc((void**)&deviceImagePtr, allocSize));
    //copy image data to GPU
    checkCudaErrors(cudaMemcpy(deviceImagePtr, hostImagePtr, allocSize, cudaMemcpyHostToDevice));
    
    //create a Image object
    return GPUImage(loadedImage.cols, loadedImage.rows, deviceImagePtr);
}


void GPUImage::saveRGB(const std::string& fileName, const GPUImage& image) {

    //load image from device to host
    const size_t allocSize = image._width * image._height * sizeof(rgbPixel);
    rgbPixel *hostImagePtr;
    checkCudaErrors(cudaMallocHost((void**)&hostImagePtr, allocSize));
    checkCudaErrors(cudaMemcpy(hostImagePtr, image._deviceImageRGBPtr, allocSize, cudaMemcpyDeviceToHost));

    //create OpenCV image
    cv::Mat cvImage(image._height, image._width, CV_8UC3, (void*)hostImagePtr);

    //write image to file
    cv::imwrite(fileName.c_str(), cvImage);
}

void GPUImage::saveHDR(const std::string& fileName, const GPUImage& image) {

    //load image from device to host
    const size_t allocSize = image._width * image._height * hdrChannelsNumber * sizeof(hdrPixel);
    hdrPixel* hostImagePtr;
    checkCudaErrors(cudaMallocHost((void**)&hostImagePtr, allocSize));
    checkCudaErrors(cudaMemcpy(hostImagePtr, image._deviceImageHDRPtr, allocSize, cudaMemcpyDeviceToHost));

	int sizes[2];
	sizes[0] = image._height;
	sizes[1] = image._width;

	cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)hostImagePtr);

	imageHDR = imageHDR * 255;

	cv::imwrite(fileName.c_str(), imageHDR);
}

GPUImage GPUImage::createEmptyRGB(int width, int height) {
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

GPUImage GPUImage::createEmptyHDR(int width, int height) {
    if (width <= 0 || height <= 0) {
        std::cerr << "Non-positive image bounds!" << std::endl;
        exit(1);
    }

    hdrPixel* devicePtr = 0;
    const size_t allocSize = width * height * hdrChannelsNumber * sizeof(hdrPixel);
    checkCudaErrors(cudaMalloc((void**)&devicePtr, allocSize));
    checkCudaErrors(cudaMemset(devicePtr, 0, allocSize));

    return GPUImage(width, height, devicePtr);
}
