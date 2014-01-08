#ifndef __TCS_CUDA_PROJ_H_INCLUDED_
#define __TCS_CUDA_PROJ_H_INCLUDED_

// -------------------------------
// ----- S T R U C T U R E S -----
// -------------------------------

#include "GPUImage.h"
using namespace tcs_cuda;

// -------------------------------
// -------- F I L T E R S --------
// -------------------------------

// --------------------
// - Czarnowicz Jakub -
// --------------------

// ----------------
// - Kolacz Piotr -
// ----------------

void projGreyscale(const GPUImage& input, GPUImage& output);

#include "projToneMapping.h"

// ----------------
// - Mikos Patryk -
// ----------------

void projInvert(const GPUImage& input, GPUImage& output);
#include "projMatrix3x3.h"
#include "projHistogram.h"
#include "projTransform.h"

#endif
