NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/share/
OPENCV_INCLUDEPATH=/usr/include/opencv2/

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

# or if using MacPorts

#OPENCV_LIBPATH=/opt/local/lib
#OPENCV_INCLUDEPATH=/opt/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

TARGET = tcs.x

CUDA_CLASSES = projInvert.cu \
			   projMatrix3x3.cu \
			   projHistogram.cu \
			   projGrayScale.cu \
			   projTransform.cu \
                           projToneMapping.cu

OBJS = projInvert.o \
	   projMatrix3x3.o \
	   projMatrix3x3Adapter.o \
	   projHistogram.o \
	   projGrayScale.o \
	   projTransform.o \
           projToneMapping.o

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCC_OPTS)

$(TARGET): compileStructs compileMatrixFilterAdapter compileMain $(OBJS)
	$(NVCC) structs.o main.o $(OBJS) -o $(TARGET) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

compileMatrixFilterAdapter: compileStructs projMatrix3x3.cpp
	g++ -c projMatrix3x3.cpp -o projMatrix3x3Adapter.o $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

compileMain: main.cpp utils.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

compileStructs: GPUImage.cpp
	g++ -c GPUImage.cpp -o structs.o $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

clean:
	rm -f *.o *.png hw
