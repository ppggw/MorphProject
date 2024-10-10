#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"

#include "cuda_header.h"

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


__global__ void RingFilterGPU(unsigned char* ImageData, unsigned char* d_ResultImage, int rows,
                                int cols, int pud, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (
        x > cols - pud - 1 ||
        y > rows - pud - 1 ||
        x < pud + 1 ||
        y < pud + 1
    )
    {
        d_ResultImage[y * cols + x] = (unsigned char)0;
        return;
    }

    float M = 0;
    for(int i = -pud; i != pud+1; i++){
        M += (int)ImageData[ (y-pud)*cols + x + i];
        M += (int)ImageData[ (y+pud)*cols + x + i];
    }

    for(int i = -pud+1; i != pud; i++){
        M += (int)ImageData[ (y+i)*cols + x - pud];
        M += (int)ImageData[ (y+i)*cols + x + pud];
    }

    M = M/( (4*pud + 2) + (4*pud - 4) );

    if(abs(ImageData[y * cols + x] - M) >= threshold){
        d_ResultImage[y * cols + x] = (unsigned char)255;
    }
    else{
        d_ResultImage[y * cols + x] = (unsigned char)0;
    }
}


extern "C" unsigned char* GPUCalc(unsigned char* ImageData, int rows, int cols, int pud, int threshold){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //есть ли в этом смысл?
//    cudaStream_t streams[NUM_STREAMS];
//    for (int i = 0; i < NUM_STREAMS; i++) { cudaStreamCreate(&streams[i]); }

    unsigned char *dev_Image;
    unsigned char *dev_Result_Image;

    cudaMalloc((void**)&dev_Image, sizeof(unsigned char) * cols * rows);
    cudaMalloc((void**)&dev_Result_Image, sizeof(unsigned char) * cols * rows);

    cudaMemcpy(dev_Image, ImageData, cols*rows * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
    dim3 blockSize (THREAD_DIM, THREAD_DIM);

    RingFilterGPU<<<gridSize, blockSize>>>(
                dev_Image,
                dev_Result_Image,
                rows,
                cols,
                pud,
                threshold
            );

    unsigned char* ResultImage = (unsigned char*)malloc(cols * rows * sizeof(unsigned char));
    cudaMemcpy(ResultImage, dev_Result_Image, sizeof(unsigned char) * cols * rows, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time of CUDA Work: %3.1f ms\n", elapsedTime);

    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

//    free(ResultImage);
    cudaFree(dev_Result_Image);
    cudaFree(dev_Image);

    return ResultImage;
}


