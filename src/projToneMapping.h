#ifndef TONEMAPPING_H
#define TONEMAPPING_H

#pragma once

#include "GPUImage.h"
using namespace tcs_cuda;

void projTonemapping(const GPUImage& input, GPUImage& output);
GPUImage hdrTransform(const GPUImage& input);
#endif /* TONEMAPPING_H */