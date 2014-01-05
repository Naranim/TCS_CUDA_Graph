#ifndef __TCS_CUDA_PROJ_HISTOGRAM_H_INCLUDED__
#define __TCS_CUDA_PROJ_HISTOGRAM_H_INCLUDED__

#pragma once

#include "GPUImage.h"
using namespace tcs_cuda;

GPUImage histogramTransform(const GPUImage& input);

void projHistogram_All(const GPUImage& input, GPUImage& output);
void projHistogram_Red(const GPUImage& input, GPUImage& output);
void projHistogram_Green(const GPUImage& input, GPUImage& output);
void projHistogram_Blue(const GPUImage& input, GPUImage& output);

#endif
