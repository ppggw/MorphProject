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


__global__ void RingFilterGPU(unsigned char* ImageData, int* vectorX, int* vectorY,
                              int* counter, int rows, int cols, int pud, int threshold)
{
    __shared__ int block_count;
    __shared__ int block_mutex;
    __shared__ int block_vectorX[THREAD_DIM * THREAD_DIM];
    __shared__ int block_vectorY[THREAD_DIM * THREAD_DIM];
    __shared__ int im_values[THREAD_DIM * THREAD_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (
        x > cols - pud - 1 ||
        y > rows - pud - 1 ||
        x < pud + 1 ||
        y < pud + 1
    )
    {
        return;
    }
    //инициализация счетчика
    if(threadIdx.x == 0){
        block_count = 0;
        block_mutex = 0;
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
//        bool blocked = true;
//           while(blocked) {
//               if(0 == atomicCAS(&mLock, 0, 1)) {
//                   vectorX[*counter] = x;
//                   vectorY[*counter] = y;
//                   *counter+=1;
//                   atomicExch(&mLock, 0);
//                   blocked = false;
//               }
//        }

        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&block_mutex, 0, 1)) {
                atomicExch(&block_vectorX[block_count], x);
                atomicExch(&block_vectorY[block_count], y);
                atomicExch(&im_values[block_count], ImageData[y * cols + x]);
                atomicAdd(&block_count, 1);

                atomicExch(&block_mutex, 0);
                blocked = false;
            }
        }
    }
    __syncthreads();
    //дальше пусть каждая стартовая нить блока обработает массив и найдем максимум
    if(threadIdx.x == 0 && threadIdx.y == 0 && block_count != 0){
        int max = im_values[0];
        int index = 0;
        for(int i = 1; i != block_count; i++){
            if (im_values[i] > max){
                max = im_values[i];
                index = i;
            }
        }
        //запись
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mLock, 0, 1)) {
                vectorX[*counter] = block_vectorX[index];
                vectorY[*counter] = block_vectorY[index];
                *counter+=1;
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
    int* DevCounter;

    int state = 0;

    cudaMalloc((void**)&dev_Image, sizeof(unsigned char) * cols * rows);
    cudaMalloc((void**)&dev_X, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&DevCounter, sizeof(int));

    cudaMemcpy(dev_Image, ImageData, cols*rows * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(DevCounter, &state, sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
    dim3 blockSize (THREAD_DIM, THREAD_DIM);

    RingFilterGPU<<<gridSize, blockSize>>>(
                dev_Image,
                dev_X,
                dev_Y,
                DevCounter,
                rows,
                cols,
                pud,
                threshold
            );
    cudaDeviceSynchronize();

    int* res_X = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* res_Y = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* counter = (int*)malloc(sizeof(int));
    cudaError_t error1 = cudaMemcpy(res_X, dev_X, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error2 = cudaMemcpy(res_Y, dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error3 = cudaMemcpy(counter, DevCounter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

//    for(int i =0; i != *counter; i++){
//        std::cout << res_X[i] << " " << res_Y[i] << "\n";
//    }

    ContForPoints* cont = (ContForPoints*)malloc(sizeof(ContForPoints));
    cont->vectorX = res_X;
    cont->vectorY = res_Y;
    cont->counter = counter;

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
    cudaFree(DevCounter);

    return cont;
}

