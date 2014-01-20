#ifndef RESCALE_H
#define RESCALE_H

#pragma once

#include "GPUImage.h"
using namespace tcs_cuda;

void rescale(const GPUImage& input, GPUImage& output);
GPUImage rescaleTransform(const GPUImage& input);

#endif