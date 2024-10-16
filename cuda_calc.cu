#include <stdio.h>
#include <math.h>
#include "cuda_header.h"
#include "cuda_runtime.h"

#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


__device__ int mLock = 0;

//__device__ int counter_for_each_block[16][20];

//__device__ int mutex_for_each_block[16][20];


__global__ void RingFilterGPU(unsigned char* ImageData, int* vectorX, int* vectorY,
                              int* counter_for_each_block, int * mutex_for_each_block,
                              int* global_counter, int rows, int cols, int pud, int threshold)
{
    __shared__ int block_vectorX[THREAD_DIM * THREAD_DIM];
    __shared__ int block_vectorY[THREAD_DIM * THREAD_DIM];
    __shared__ int im_values[THREAD_DIM * THREAD_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (
        x > cols - pud - 5 ||
        y > rows - pud - 5 ||
        x < pud + 5 ||
        y < pud + 5
    )
    {
        return;
    }

    if(threadIdx.x == 0){
        counter_for_each_block[blockIdx.y + blockIdx.x] = 0;
        mutex_for_each_block[blockIdx.y + blockIdx.x] = 0;
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
//        atomicExch(&block_vectorX[ counter_for_each_block[blockIdx.y + blockIdx.x] ], x);
//        atomicExch(&block_vectorY[ counter_for_each_block[blockIdx.y + blockIdx.x] ], y);
//        atomicExch(&im_values[ counter_for_each_block[blockIdx.y + blockIdx.x] ], ImageData[y * cols + x]);
//        atomicAdd(&counter_for_each_block[blockIdx.y + blockIdx.x], 1);
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mutex_for_each_block[blockIdx.y + blockIdx.x], 0, 1)){
                atomicExch(&block_vectorX[ counter_for_each_block[blockIdx.y + blockIdx.x] ], x);
                atomicExch(&block_vectorY[ counter_for_each_block[blockIdx.y + blockIdx.x] ], y);
                atomicExch(&im_values[ counter_for_each_block[blockIdx.y + blockIdx.x] ], ImageData[y * cols + x]);
                atomicAdd(&counter_for_each_block[blockIdx.y + blockIdx.x], 1);

                atomicExch(&mutex_for_each_block[blockIdx.y + blockIdx.x], 0);
                blocked = false;
            }
        }
    }
    __syncthreads();
    //дальше пусть каждая стартовая нить блока обработает массив и найдем максимум
    if(threadIdx.x == 0 && threadIdx.y == 0 && counter_for_each_block[blockIdx.y + blockIdx.x] != 0){
        int max = im_values[0];
        int index = 0;
        for(int i = 1; i != counter_for_each_block[blockIdx.y + blockIdx.x]; i++){
            if (im_values[i] > max){
                max = im_values[i];
                index = i;
            }
        }
        //запись
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mLock, 0, 1)) {
                vectorX[*global_counter] = block_vectorX[index];
                vectorY[*global_counter] = block_vectorY[index];
                *global_counter+=1;
                atomicExch(&mLock, 0);
                blocked = false;
            }
        }
    }
    __syncthreads();
}


extern "C" ContForPoints* GPUCalc(unsigned char* ImageData, int rows, int cols, int pud, int threshold){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //есть ли в этом смысл?
//    cudaStream_t streams[NUM_STREAMS];
//    for (int i = 0; i < NUM_STREAMS; i++) { cudaStreamCreate(&streams[i]); }

    unsigned char *dev_Image;
    int* dev_X;
    int* dev_Y;
    int* devCounterGlobal;

    int* devCounterForEachBlock;
    int* devMutexForEachBlock;

    int state = 0;

    cudaMalloc((void**)&dev_Image, sizeof(unsigned char) * cols * rows);
    cudaMalloc((void**)&dev_X, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&devCounterGlobal, sizeof(int));

    cudaMalloc((void**)&devCounterForEachBlock, sizeof(int) * 16 * 20);
    cudaMalloc((void**)&devMutexForEachBlock, sizeof(int) * 16 * 20);

    cudaMemcpy(dev_Image, ImageData, cols*rows * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(devCounterGlobal, &state, sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM)); // x = 20 y = 16
    dim3 blockSize (THREAD_DIM, THREAD_DIM);

    RingFilterGPU<<<gridSize, blockSize>>>(
                dev_Image,
                dev_X,
                dev_Y,
                devCounterForEachBlock,
                devMutexForEachBlock,
                devCounterGlobal,
                rows,
                cols,
                pud,
                threshold
            );
    cudaDeviceSynchronize();

    int* res_X = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* res_Y = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* res_counter = (int*)malloc(sizeof(int));
    cudaError_t error1 = cudaMemcpy(res_X, dev_X, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error2 = cudaMemcpy(res_Y, dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error3 = cudaMemcpy(res_counter, devCounterGlobal, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

//    for(int i =0; i != *counter; i++){
//        std::cout << res_X[i] << " " << res_Y[i] << "\n";
//    }

    ContForPoints* cont = (ContForPoints*)malloc(sizeof(ContForPoints));
    cont->vectorX = res_X;
    cont->vectorY = res_Y;
    cont->counter = res_counter;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time of CUDA Work: %3.1f ms\n", elapsedTime);

    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    cudaFree(dev_Image);
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(devCounterGlobal);
    cudaFree(devCounterForEachBlock);
    cudaFree(devMutexForEachBlock);

    return cont;
}

