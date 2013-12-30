#include "projMatrix3x3.h"

// ------------
// - Identity -
// ------------
void projMatrix3x3_Identity(const GPUImage& input, GPUImage& output) {
    int matrix[9] = { 
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}

// ---------------------------
// - Edge detection standard -
// ---------------------------
void projMatrix3x3_EdgeDetection_Vertical(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, 0, 0,
        -1, 1, 0,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_Horizontal(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_Diagonal1(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_Diagonal2(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, 0, -1,
        0, 1, 0,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}

// ------------------------
// - Edge detection Sobel -
// ------------------------
void projMatrix3x3_EdgeDetection_SobelVertical(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_SobelHorizontal(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };
    projMatrix3x3(input, output, matrix);
}

// --------------------------
// - Edge detection Prewitt -
// --------------------------
void projMatrix3x3_EdgeDetection_PrewittVertical(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_PrewittHorizontal(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        0, 0, 0,
        -1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}

// --------------------------
// - Edge detection Laplace -
// --------------------------
void projMatrix3x3_EdgeDetection_Laplace1(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_Laplace2(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_Laplace3(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, -2, 1,
        -2, 4, -2,
        1, -2, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_LaplaceVertical(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, 0, 0,
        -1, 2, -1,
        0, 0, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_LaplaceHorizontal(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, 0,
        0, 2, 0,
        0, -1, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_LaplaceDiagonal(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, 0, -1,
        0, 4, 0,
        -1, 0, -1
    };
    projMatrix3x3(input, output, matrix);
}

// ---------------------------
// - Edge detection gradient - 
// ---------------------------
void projMatrix3x3_EdgeDetection_GradientEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, 1, 1,
        -1, -2, 1,
        -1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientSouthEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, 1,
        -1, -2, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientSouth(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, -1,
        1, -2, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientSouthWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, -1, -1,
        1, -2, -1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, -1,
        1, -2, -1,
        1, 1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientNorthWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, -2, -1,
        1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientNorth(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, -2, 1,
        -1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EdgeDetection_GradientNorthEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        -1, -2, 1,
        -1, -1, 1
    };
    projMatrix3x3(input, output, matrix);
}

// -------------------
// - LowPass filters -
// -------------------
void projMatrix3x3_Average(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_LowPass1(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, 2, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_LowPass2(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, 4, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_LowPass3(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        1, 12, 1,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_LowPassGauss(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    projMatrix3x3(input, output, matrix);
}

// --------------------
// - HighPass filters -
// --------------------
void projMatrix3x3_MeanRemoval(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_HighPass1(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_HighPass2(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, -2, 1,
        -2, 5, -2,
        1, -2, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_HighPass3(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, 0,
        -1, 20, -1,
        0, -1, 0
    };
    projMatrix3x3(input, output, matrix);
}

// ---------
// - Embos -
// ---------
void projMatrix3x3_EmbosEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, 0, 1,
        -1, 1, 1,
        -1, 0, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosSouthEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, 0,
        -1, 1, 1,
        0, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosSouth(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        -1, -1, -1,
        0, 1, 0,
        1, 1, 1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosSouthWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, -1, -1,
        1, 1, -1,
        1, 1, 0
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 0, -1,
        1, 1, -1,
        1, 0, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosNorthWest(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 0,
        1, 1, -1,
        0, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosNorth(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        1, 1, 1,
        0, 1, 0,
        -1, -1, -1
    };
    projMatrix3x3(input, output, matrix);
}
void projMatrix3x3_EmbosNorthEast(const GPUImage& input, GPUImage& output){
    int matrix[9] = { 
        0, 1, 1,
        -1, 1, 1,
        -1, -1, 0
    };
    projMatrix3x3(input, output, matrix);
}
