#include "utils.h"
#include <cmath>
#include <stdio.h>
#include "GPUImage.h"
#include <string>
#include <thrust/extrema.h>
#include "projToneMapping.h"
using namespace tcs_cuda;

__global__ 
void rgb_to_xyY(float* d_r, float* d_g, float* d_b, float* d_x, float* d_y,
			float* d_log_Y, float delta, int num_pixels_y, int num_pixels_x){
			
	int ny = num_pixels_y;
	int nx = num_pixels_x;
	int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
	int image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

	if (image_index_2d.x < nx && image_index_2d.y < ny){
		float r = d_r[ image_index_1d ];
		float g = d_g[ image_index_1d ];
		float b = d_b[ image_index_1d ];

		float X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
		float Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
		float Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );

		float L = X + Y + Z;
		float x = X / L;
		float y = Y / L;

		float log_Y = log10f( delta + Y );

		d_x[ image_index_1d ] = x;
		d_y[ image_index_1d ] = y;
		d_log_Y[ image_index_1d ] = log_Y;
	}
}

__global__ 
void normalize_cdf(unsigned int* d_input_cdf, float* d_output_cdf, int n){

	const float normalization_constant = 1.f / d_input_cdf[n - 1];

	int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	if ( global_index_1d < n ){
		unsigned int input_value  = d_input_cdf[ global_index_1d ];
		float output_value = input_value * normalization_constant;
		d_output_cdf[ global_index_1d ] = output_value;
	}
}

__global__ 
void tonemap(float* d_x, float* d_y, float* d_log_Y, float* d_cdf_norm,
			float* d_r_new, float* d_g_new, float* d_b_new, float min_log_Y,
			float max_log_Y, int num_bins, int num_pixels_y, int num_pixels_x){
			
  float log_Y_range = max_log_Y - min_log_Y;
	int ny = num_pixels_y;
	int nx = num_pixels_x;
	int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
	int image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

	if ( image_index_2d.x < nx && image_index_2d.y < ny ){
		float x = d_x[ image_index_1d ];
		float y = d_y[ image_index_1d ];
		float log_Y = d_log_Y[ image_index_1d ];
		int bin_index = min( num_bins - 1, int( (num_bins * ( log_Y - min_log_Y ) ) / log_Y_range ) );
		float Y_new = d_cdf_norm[ bin_index ];

		float X_new = x * ( Y_new / y );
		float Z_new = ( 1 - x - y ) * ( Y_new / y );

		float r_new = ( X_new *  3.2406f ) + ( Y_new * -1.5372f ) + ( Z_new * -0.4986f );
		float g_new = ( X_new * -0.9689f ) + ( Y_new *  1.8758f ) + ( Z_new *  0.0415f );
		float b_new = ( X_new *  0.0557f ) + ( Y_new * -0.2040f ) + ( Z_new *  1.0570f );

		d_r_new[ image_index_1d ] = r_new;
		d_g_new[ image_index_1d ] = g_new;
		d_b_new[ image_index_1d ] = b_new;
	}
}

__global__
void init_array(const size_t numBins, unsigned int* array){
    int blockSize = 1024;
    int idx = threadIdx.x + blockSize * blockIdx.x;
    if(idx < numBins){
        array[idx] = 0;
    }
}

__global__
void create_histogram(const float* const d_logLuminance, float min_logLum, const size_t numBins, float logLumRange, unsigned int* d_bins){
    int blockSize = 1024;
    int idx = threadIdx.x + blockSize * blockIdx.x;
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_logLuminance[idx] - min_logLum) / logLumRange * numBins));
    atomicAdd(&(d_bins[bin]),1);
}

__global__
void find_min(const float* const inputMin, float* minLuminance, int n){
    int blockSize = blockDim.x;
    int threadId = threadIdx.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
    int firstIdx = blockSize * blockIdx.x;
    
    extern __shared__ float sdata[];
    
    if(idx < n){
        sdata[threadId] = inputMin[idx];
    }else{
        sdata[threadId] = inputMin[firstIdx];
    }
      
    __syncthreads();
    for(int s = blockSize/2; s > 0; s >>=1){
        if(threadId < s){
            sdata[threadId] = min(sdata[threadId], sdata[threadId + s]);
        }
        __syncthreads();
    }
    
    if(threadId == 0){
        minLuminance[blockIdx.x] = sdata[0];
    }
}

__global__
void find_max(const float* const inputMax, float* maxLuminance, int n){
    int blockSize = blockDim.x;
    int threadId = threadIdx.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
    int firstIdx = blockSize * blockIdx.x;
    
    extern __shared__ float sdata[];
    
    if(idx < n){
        sdata[threadId] = inputMax[idx];
    }else{
        sdata[threadId] = inputMax[firstIdx];
    }
      
    __syncthreads();
    for(int s = blockSize/2; s > 0; s >>=1){
        if(threadId < s){
            sdata[threadId] = max(sdata[threadId], sdata[threadId + s]);
        }
        __syncthreads();
    }
    
    if(threadId == 0){
        maxLuminance[blockIdx.x] = sdata[0];
    }
}

__global__
void prefix_sum(unsigned int* input, unsigned int* block_sum, const size_t n){
    int blockSize = 1024;
    int threadId = threadIdx.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
	
    __shared__ float sdata[1024];
	if(idx < n && threadId > 0){
		sdata[threadId] = input[idx-1]; 
	}else{
		sdata[threadId] = 0;
	}

	__syncthreads();

	//prefix sum
	for (int shift = 1; shift < blockSize; shift <<= 1){
		if (threadId < shift){
			continue;
		}
		
		int tmp = sdata[threadId-shift];
		
		__syncthreads();
		
		sdata[threadId] += tmp ;	
		
		__syncthreads();
	}

    if(threadId + 1 == blockSize){//ostatni element w bloku
        int last = input[idx];
        for(int blockNo = blockIdx.x + 1; blockNo < gridDim.x; ++blockNo){
            atomicAdd(&(block_sum[blockNo]),sdata[threadId] + last);
        }
	}
	
	if(threadId < n){
		input[idx] = sdata[threadId];
	}
	__syncthreads();
}    

__global__
void create_cdf(unsigned int* input, unsigned int* block_sum, const size_t n, unsigned int* const d_cdf){
    int blockSize = 1024;
    int threadId = threadIdx.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
	
    __shared__ int add;
    if(threadId == 0){
        if(blockIdx.x == 0){
            add = 0;
        }else{
            add = block_sum[blockIdx.x];
        }
    }
    
    __syncthreads();
    
	if(idx < n){
        d_cdf[idx] = add + input[idx];
	}
}    

__global__
void split_to_channels(float* imgPtr, float* d_red, float* d_green, float* d_blue, int numPixels){
	int idX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = gridDim.x * blockDim.x * idY + idX;

	if(idx < numPixels){
		d_blue[idx]  = imgPtr[3 * idx + 0];
		d_green[idx] = imgPtr[3 * idx + 1];
		d_red[idx]   = imgPtr[3 * idx + 2];			
	}
}

__global__
void recombine_channels(float* imgPtr, float* d_red, float* d_green, float* d_blue, int numPixels){
	int idX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = gridDim.x * blockDim.x * idY + idX;

	if(idx < numPixels){
		imgPtr[3 * idx + 0] = d_blue[idx];
		imgPtr[3 * idx + 1] = d_green[idx];
		imgPtr[3 * idx + 2] = d_red[idx];	
	}
}

void calculate_histogram_and_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	int totalSize = numRows * numCols;  

	//Step 1
	int blockSize = 1024;  
	int blocksNum = totalSize;
	float* d_output;  
	const float* d_input = d_logLuminance;  
	int n = totalSize;
	cudaMalloc((void**) &d_output, sizeof(float) * blocksNum);
	do{  
		blocksNum = ceil(blocksNum / (blockSize * 1.0));  
		find_min<<<blocksNum,blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);
		d_input = d_output;      
		n = blocksNum;
	}while(blocksNum > 1);
	cudaMemcpy(&min_logLum,d_output,sizeof(float), cudaMemcpyDeviceToHost);
	blocksNum = totalSize;
	d_input = d_logLuminance;  
	n = totalSize;
	do{  
		blocksNum = ceil(blocksNum / (blockSize * 1.0));  
		find_max<<<blocksNum,blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);
		d_input = d_output;      
		n = blocksNum;
	}while(blocksNum > 1);
	cudaMemcpy(&max_logLum,d_output,sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree((void *)d_output);

	//Step 2
	float logLumRange = max_logLum - min_logLum;

	//Step 3
	unsigned int *d_bins;
	cudaMalloc((void**) &d_bins, sizeof(unsigned int) * numBins);
	blocksNum = ceil(numBins / (blockSize * 1.0));    
	init_array<<<blocksNum,blockSize>>>(numBins, d_bins);
	blocksNum = ceil(totalSize / (blockSize * 1.0));    
	create_histogram<<<blocksNum,blockSize>>>(d_logLuminance, min_logLum, numBins, logLumRange, d_bins);


	//Step 4
	unsigned int *block_sum;
	cudaMalloc((void**) &block_sum, sizeof(unsigned int) * numBins);
	blocksNum = ceil(numBins / (blockSize * 1.0));    
	init_array<<<blocksNum,blockSize>>>(numBins, block_sum);
	prefix_sum<<<blocksNum,blockSize>>>(d_bins, block_sum, numBins);
	create_cdf<<<blocksNum,blockSize>>>(d_bins, block_sum, numBins, d_cdf);  

	cudaFree((void *)d_bins);
	cudaFree((void *)block_sum);
}

GPUImage hdrTransform(const GPUImage& input) {
    return GPUImage::createEmptyHDR(input.getWidth(), input.getHeight());
}

void projTonemapping(const GPUImage& input, GPUImage& output){
	int numBins = 1024;
	unsigned int* d_cdf;
	size_t cdf_size = sizeof(unsigned int) * numBins;
	checkCudaErrors(cudaMalloc(&d_cdf, cdf_size));
	checkCudaErrors(cudaMemset(d_cdf, 0, cdf_size));

	size_t numRows = input.getHeight();
	size_t numCols = input.getWidth();

	//first thing to do is split incoming BGR float data into separate channels
	size_t numPixels = numRows * numCols;
	int blockX = 32;
	int blockY = 16;
	int blockZ = 1;
	const dim3 blockSize(blockX, blockY, blockZ);
	
	int gridX = (numCols + blockSize.x - 1) / blockSize.x;//ceil(numCols / (blockX * 1.0));  
	int gridY = (numRows + blockSize.y - 1) / blockSize.y;//ceil(numRows / (blockY * 1.0));  
	int gridZ = 1;
	const dim3 gridSize(gridX, gridY, gridZ);

	//RGB space
	float *d_red, *d_green, *d_blue;  

	size_t channelSize = sizeof(float) * numPixels;

	checkCudaErrors(cudaMalloc(&d_red, channelSize));
	checkCudaErrors(cudaMalloc(&d_green, channelSize));
	checkCudaErrors(cudaMalloc(&d_blue, channelSize));

	split_to_channels<<<gridSize, blockSize>>>(input.getDeviceHDRPixels(), d_red, d_green, d_blue, numPixels);

	//chroma-LogLuminance Space
	float *d_x, *d_y, *d_luminance;

	checkCudaErrors(cudaMalloc(&d_x, channelSize));
	checkCudaErrors(cudaMalloc(&d_y, channelSize));
	checkCudaErrors(cudaMalloc(&d_luminance, channelSize));

	//convert from RGB space to chrominance/luminance space xyY
	rgb_to_xyY<<<gridSize, blockSize>>>(d_red, d_green, d_blue,d_x, d_y, d_luminance,
										.0001f, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	float min_logLum = 0.f;
	float max_logLum = 1.f;
	//call the students' code
	calculate_histogram_and_cdf(d_luminance, d_cdf, min_logLum, max_logLum, numRows, numCols, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//check results and output the tone-mapped image
	const int numThreads = 192;

	float *d_cdf_normalized;

	checkCudaErrors(cudaMalloc(&d_cdf_normalized, sizeof(float) * numBins));

	//first normalize the cdf to a maximum value of 1
	//this is how we compress the range of the luminance channel
	normalize_cdf<<< (numBins + numThreads - 1) / numThreads,
					numThreads>>>(d_cdf, d_cdf_normalized, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//next perform the actual tone-mapping
	//we map each luminance value to its new value
	//and then transform back to RGB space
	tonemap<<<gridSize, blockSize>>>(d_x, d_y, d_luminance, d_cdf_normalized,
								d_red, d_green, d_blue, min_logLum, max_logLum,
								numBins, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	recombine_channels<<<gridSize, blockSize>>>(output.getDeviceHDRPixels(), d_red, d_green, d_blue, numPixels);

	//cleanup
	checkCudaErrors(cudaFree(d_cdf));
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_luminance));
	checkCudaErrors(cudaFree(d_cdf_normalized));
}
