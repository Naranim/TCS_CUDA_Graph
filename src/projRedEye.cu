#include "utils.h"
#include <cmath>
#include <stdio.h>
#include "GPUImage.h"
#include <string>
#include <thrust/host_vector.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include "projRedEye.h"
using namespace tcs_cuda;

//simple cross correlation kernel copied from Mike's IPython Notebook
__global__ 
void naive_normalized_cross_correlation(float* d_response, unsigned char* d_original,
		unsigned char* d_template, int num_pixels_y, int num_pixels_x, int template_half_height,
		int template_height, int template_half_width, int template_width, float template_mean){
	
	int ny = num_pixels_y;
	int nx = num_pixels_x;
	int knx = template_width;
	int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
	int image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

	if ( image_index_2d.x < nx && image_index_2d.y < ny ){
		int template_size = template_width * template_height;
		//
		// compute image mean
		//
		float image_sum = 0.0f;

		for ( int y = -template_half_height; y <= template_half_height; y++ ){
			for ( int x = -template_half_width; x <= template_half_width; x++ ){
				int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
				int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
				int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

				unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];

				image_sum += (float)image_offset_value;
			}
		}

		float image_mean = image_sum / (float)template_size;

		//
		// compute sums
		//
		float sum_of_image_template_diff_products = 0.0f;
		float sum_of_squared_image_diffs          = 0.0f;
		float sum_of_squared_template_diffs       = 0.0f;

		for ( int y = -template_half_height; y <= template_half_height; y++ ){
			  for ( int x = -template_half_width; x <= template_half_width; x++ ){
				int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
				int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
				int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

				unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];
				float         image_diff         = (float)image_offset_value - image_mean;

				int2 template_index_2d = make_int2( x + template_half_width, y + template_half_height );
				int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

				unsigned char template_value = d_template[ template_index_1d ];
				float         template_diff  = template_value - template_mean;

				float image_template_diff_product = image_offset_value   * template_diff;
				float squared_image_diff          = image_diff           * image_diff;
				float squared_template_diff       = template_diff        * template_diff;

				sum_of_image_template_diff_products += image_template_diff_product;
				sum_of_squared_image_diffs          += squared_image_diff;
				sum_of_squared_template_diffs       += squared_template_diff;
			  }
		}


		//
		// compute final result
		//
		float result_value = 0.0f;

		if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 ){
			result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
		}

		d_response[ image_index_1d ] = result_value;
	}
}


__global__ 
void remove_redness_from_coordinates(const unsigned int* d_coordinates, unsigned char* d_r,
			unsigned char* d_b, unsigned char* d_g, unsigned char* d_r_output, int num_coordinates,
			int num_pixels_y, int num_pixels_x, int template_half_height, int template_half_width){
			
	int ny  = num_pixels_y;
	int nx = num_pixels_x;
	int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	int imgSize = num_pixels_x * num_pixels_y;

	if ( global_index_1d < num_coordinates ){
		unsigned int image_index_1d = d_coordinates[ imgSize - global_index_1d - 1 ];
		ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

		for ( int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++ ){
			for ( int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++ ){
			int2 image_offset_index_2d         = make_int2( x, y );
			int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
			int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

			unsigned char g_value = d_g[ image_offset_index_1d_clamped ];
			unsigned char b_value = d_b[ image_offset_index_1d_clamped ];

			unsigned int gb_average = ( g_value + b_value ) / 2;

			d_r_output[ image_offset_index_1d_clamped ] = (unsigned char)gb_average;
			}
		}
	}
}


struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> >{
	__host__ __device__
	thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel){
		return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
	}
};

struct combineChannels : thrust::unary_function<thrust::tuple<unsigned char, unsigned char, unsigned char>, uchar4>{	
	__host__ __device__
	uchar4 operator()(thrust::tuple<unsigned char, unsigned char, unsigned char> t){
		return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
	}
};

struct combineResponses : thrust::unary_function<float, thrust::tuple<float, float, float> >{
	__host__ __device__
	float operator()(thrust::tuple<float, float, float> t){
		return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
	}
};

__global__
void init_array_red(const size_t numBins, unsigned int* array){
    int blockSize = blockDim.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
    if(idx < numBins){
        array[idx] = 0;
    }
}

__global__
void init_predicates(size_t n, unsigned int* d_predicates, unsigned int* const d_inputVals, unsigned int mask, unsigned int bit, unsigned int predicate){
    int blockSize = blockDim.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
	
	if(idx < n){
		unsigned int value = (d_inputVals[idx] & mask) >> bit;
		if(value == predicate){
			d_predicates[idx] = 1;
		}else{
			d_predicates[idx] = 0;
		}
	}
}

__global__
void create_histogram(unsigned int* const d_inputVals, unsigned int mask, unsigned int bit, size_t n, unsigned int* d_histogram){
    int blockSize = blockDim.x;
    int idx = threadIdx.x + blockSize * blockIdx.x;
	
	if(idx < n){
		unsigned int bin = (d_inputVals[idx] & mask) >> bit;
		atomicAdd(&(d_histogram[bin]),1);
	}
}

__global__
void prefix_sum_red(unsigned int* input, unsigned int* block_sum, const size_t n){
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
void compact(unsigned int* input, unsigned int* block_sum, const size_t n, unsigned int* const output){
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
        output[idx] = add + input[idx];
	}
}    

__global__
void place_elements(unsigned int* const d_positions, unsigned int* const d_histogram, unsigned int predicate, unsigned int* const d_inputVals,unsigned int* const d_inputPos,unsigned int* const d_outputVals,unsigned int* const d_outputPos, size_t n, unsigned int mask, unsigned int bit){
    int blockSize = 1024;
    int idx = threadIdx.x + blockSize * blockIdx.x;
	
	__shared__ unsigned int prefix;
	
	if(threadIdx.x == 0){
		if(predicate > 0){
			prefix = d_histogram[predicate-1];
		}else{
			prefix = 0;
		}
	}
	
	__syncthreads();
	
	if(idx < n){
		unsigned int value = d_inputVals[idx];
		unsigned int bin = (value & mask) >> bit;
		if(bin == predicate){
			unsigned int pos = prefix + d_positions[idx];
			d_outputVals[pos] = value;
			d_outputPos[pos]  = d_inputPos[idx];
		}
	}
}

void sort_scored_values_and_positions(unsigned int* const d_inputVals,
        unsigned int* const d_inputPos, unsigned int* const d_outputVals,
        unsigned int* const d_outputPos, const size_t numElems){
		
	int blockSize = 1024;  
	int blocksNum;
    int numBits = 1;
    int numBins = 1 << numBits;
    
	unsigned int *d_histogram;
	cudaMalloc((void**) &d_histogram, sizeof(unsigned int) * numBins);
	unsigned int *d_predicates;
	cudaMalloc((void**) &d_predicates, sizeof(unsigned int) * numElems);
	unsigned int *d_positions;
	cudaMalloc((void**) &d_positions, sizeof(unsigned int) * numElems);
	unsigned int *block_sum;
	cudaMalloc((void**) &block_sum, sizeof(unsigned int) * numElems);

	unsigned int *vals_src = d_inputVals;
	unsigned int *pos_src  = d_inputPos;

	unsigned int *vals_dst = d_outputVals;
	unsigned int *pos_dst  = d_outputPos;

    for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
		unsigned int mask = (numBins - 1) << i;
		blocksNum = ceil(numBins / (blockSize * 1.0));    
		init_array_red<<<blocksNum,blockSize>>>(numBins, d_histogram);
		blocksNum = ceil(numElems / (blockSize * 1.0));    
		create_histogram<<<blocksNum,blockSize>>>(vals_src, mask, i, numElems, d_histogram);
					
		blocksNum = ceil(numElems / (blockSize * 1.0));    
		//compact zeros
		init_predicates<<<blocksNum,blockSize>>>(numElems, d_predicates, vals_src, mask, i, 0);
		init_array_red<<<blocksNum,blockSize>>>(numElems, block_sum);
		prefix_sum_red<<<blocksNum,blockSize>>>(d_predicates, block_sum, numElems);
		compact<<<blocksNum,blockSize>>>(d_predicates, block_sum, numElems, d_positions);  
		place_elements<<<blocksNum,blockSize>>>(d_positions, d_histogram, 0, vals_src, pos_src, vals_dst, pos_dst, numElems, mask, i);  
		
		//compact ones
		init_predicates<<<blocksNum,blockSize>>>(numElems, d_predicates, vals_src, mask, i, 1);
		init_array_red<<<blocksNum,blockSize>>>(numElems, block_sum);
		prefix_sum_red<<<blocksNum,blockSize>>>(d_predicates, block_sum, numElems);
		compact<<<blocksNum,blockSize>>>(d_predicates, block_sum, numElems, d_positions);  
		place_elements<<<blocksNum,blockSize>>>(d_positions, d_histogram, 1, vals_src, pos_src, vals_dst, pos_dst, numElems, mask, i);  

		//swap the buffers (pointers only)
		std::swap(vals_dst, vals_src);
		std::swap(pos_dst, pos_src);
	}

	//we did an even number of iterations, need to copy from input buffer into output
	cudaMemcpy(d_outputVals,d_inputVals,sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos,d_inputPos,sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);

	cudaFree(d_histogram);
	cudaFree(block_sum);
	cudaFree(d_predicates);
	cudaFree(d_positions);
}

void projRedeye(const GPUImage& input, GPUImage& output){
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	uchar4 *inImg = input.getDeviceRGBAPixels();
	GPUImage templateImage = GPUImage::loadRGBA("eye_template.jpg");
	uchar4 *eyeTemplate = templateImage.getDeviceRGBAPixels();

	size_t numRowsTemplate = templateImage.getHeight();
	size_t numColsTemplate = templateImage.getWidth();
	size_t numRowsImg = input.getHeight();
	size_t numColsImg = input.getWidth();
	size_t templateHalfWidth = (numColsTemplate - 1) / 2;
	size_t templateHalfHeight = (numRowsTemplate - 1) / 2;

	//we need to split each image into its separate channels
	//use thrust to demonstrate basic uses
	size_t numElem = input.getWidth() * input.getHeight();
	size_t templateSize = numRowsTemplate * numColsTemplate;

	thrust::device_vector<uchar4> d_Img(inImg, inImg + numRowsImg * numColsImg);
	thrust::device_vector<uchar4> d_Template(eyeTemplate, eyeTemplate + numRowsTemplate * numColsTemplate);

	thrust::device_vector<unsigned char> d_red(numElem);
	thrust::device_vector<unsigned char> d_blue(numElem);
	thrust::device_vector<unsigned char> d_green(numElem);
	thrust::device_vector<unsigned char> d_red_template(templateSize);
	thrust::device_vector<unsigned char> d_blue_template(templateSize);
	thrust::device_vector<unsigned char> d_green_template(templateSize);

	//split the image
	thrust::transform(d_Img.begin(), d_Img.end(), 
					thrust::make_zip_iterator(thrust::make_tuple(d_red.begin(), d_blue.begin(), d_green.begin())),
                    splitChannels());

	//split the template
	thrust::transform(d_Template.begin(), d_Template.end(), 
                    thrust::make_zip_iterator(thrust::make_tuple(d_red_template.begin(), d_blue_template.begin(), d_green_template.begin())),
                    splitChannels());

	thrust::device_vector<float> d_red_response(numElem);
	thrust::device_vector<float> d_blue_response(numElem);
	thrust::device_vector<float> d_green_response(numElem);

	//need to compute the mean for each template channel
	unsigned int r_sum = thrust::reduce(d_red_template.begin(), d_red_template.end(), 0);
	unsigned int b_sum = thrust::reduce(d_blue_template.begin(), d_blue_template.end(), 0);
	unsigned int g_sum = thrust::reduce(d_green_template.begin(), d_green_template.end(), 0);

	float r_mean = (double)r_sum / templateSize;
	float b_mean = (double)b_sum / templateSize;
	float g_mean = (double)g_sum / templateSize;

	const dim3 blockSize(32, 8, 1);
	const dim3 gridSize( (numColsImg + blockSize.x - 1) / blockSize.x, (numRowsImg + blockSize.y - 1) / blockSize.y, 1);

	//now compute the cross-correlations for each channel
	naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_red_response.data()),
            thrust::raw_pointer_cast(d_red.data()), thrust::raw_pointer_cast(d_red_template.data()), 
			numRowsImg, numColsImg, templateHalfHeight, numRowsTemplate, templateHalfWidth, numColsTemplate, r_mean);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
                                                             
	naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_blue_response.data()),
            thrust::raw_pointer_cast(d_blue.data()), thrust::raw_pointer_cast(d_blue_template.data()), 
			numRowsImg, numColsImg, templateHalfHeight, numRowsTemplate, templateHalfWidth, numColsTemplate, b_mean);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_green_response.data()),
            thrust::raw_pointer_cast(d_green.data()), thrust::raw_pointer_cast(d_green_template.data()),
            numRowsImg, numColsImg, templateHalfHeight, numRowsTemplate, templateHalfWidth, numColsTemplate, g_mean);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//generate combined response - multiply all channels together
	thrust::device_vector<float> d_combined_response(numElem);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_red_response.begin(), d_blue_response.begin(), d_green_response.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(d_red_response.end(), d_blue_response.end(), d_green_response.end())),
                    d_combined_response.begin(), combineResponses());

	//find max/min of response
	typedef thrust::device_vector<float>::iterator floatIt;
	thrust::pair<floatIt, floatIt> minmax = thrust::minmax_element(d_combined_response.begin(), d_combined_response.end());

	float bias = *minmax.first;
	//we need to make all the numbers positive so that we can sort them without any bit twiddling
	thrust::transform(d_combined_response.begin(), d_combined_response.end(), thrust::make_constant_iterator(-bias), 
                d_combined_response.begin(), thrust::plus<float>());

	//now we need to create the 1-D coordinates that will be attached to the keys
	thrust::device_vector<unsigned int> coords(numElem);
	thrust::sequence(coords.begin(), coords.end()); //[0, ..., numElem - 1]

	unsigned int* outputVals;
	unsigned int* outputPos;
	checkCudaErrors(cudaMalloc(&outputVals, sizeof(unsigned int) * numElem));
	checkCudaErrors(cudaMalloc(&outputPos,  sizeof(unsigned int) * numElem));
	checkCudaErrors(cudaMemset(outputVals, 0, sizeof(unsigned int) * numElem));
	checkCudaErrors(cudaMemset(outputPos, 0,  sizeof(unsigned int) * numElem));
	
	sort_scored_values_and_positions((unsigned int*)(thrust::raw_pointer_cast(d_combined_response.data())), 
			(unsigned int*)(thrust::raw_pointer_cast(coords.data())), outputVals, outputPos, numElem);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	thrust::device_vector<unsigned char> d_output_red = d_red;

	int num_coordinates_to_recolor = templateHalfHeight * 0.6 *  templateHalfWidth * 0.6;
	const dim3 blockSize1(256, 1, 1);
	const dim3 gridSize1( (40 + blockSize1.x - 1) / blockSize1.x, 1, 1);

	remove_redness_from_coordinates<<<gridSize1, blockSize1>>>(outputPos,
				thrust::raw_pointer_cast(d_red.data()), thrust::raw_pointer_cast(d_blue.data()),
				thrust::raw_pointer_cast(d_green.data()), thrust::raw_pointer_cast(d_output_red.data()),
				num_coordinates_to_recolor,/*int num_coordinates*/ numRowsImg, numColsImg, templateHalfHeight * 0.6, templateHalfWidth * 0.6);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//combine the new red channel with original blue and green for output
	thrust::device_vector<uchar4> d_outputImg(numElem);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_output_red.begin(), d_blue.begin(), d_green.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(d_output_red.end(), d_blue.end(), d_green.end())),
					d_outputImg.begin(), combineChannels());

	cudaMemcpy(output.getDeviceRGBAPixels(), thrust::raw_pointer_cast(d_outputImg.data()), sizeof(uchar4) * numElem, cudaMemcpyDeviceToDevice);

	//Clear the global vectors otherwise something goes wrong trying to free them
	d_red.clear(); d_red.shrink_to_fit();
	d_blue.clear(); d_blue.shrink_to_fit();
	d_green.clear(); d_green.shrink_to_fit();
}

GPUImage rgbaTransform(const GPUImage& input) {
    return GPUImage::createEmptyRGBA(input.getWidth(), input.getHeight());
}
