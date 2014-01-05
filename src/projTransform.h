#ifndef TRANSFORM_H
#define TRANSFORM_H

#pragma once

#include "GPUImage.h"
using namespace tcs_cuda;

void flipVer(const GPUImage& input, GPUImage& output);
void flipHor(const GPUImage& input, GPUImage& output);

#endif /* TRANSFORM_H */