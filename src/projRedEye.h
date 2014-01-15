#ifndef REDEYE_H
#define REDEYE_H

#pragma once

#include "GPUImage.h"
using namespace tcs_cuda;

void projRedeye(const GPUImage& input, GPUImage& output);
GPUImage rgbaTransform(const GPUImage& input);
#endif /* REDEYE_H */