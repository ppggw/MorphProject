#include <stdio.h>
#include <math.h>
#include "cuda_header.h"
#include "cuda_runtime.h"

#include <vector>
#include <iostream>

__device__ int mLock = 0;


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

    if(threadIdx.x == 0 && threadIdx.y == 0){
        counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] = 0;
        mutex_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] = 0;
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
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mutex_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x], 0, 1)){
                block_vectorX[ counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] ] = x;
                block_vectorY[ counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] ] = y;
                im_values[ counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] ] = ImageData[y * cols + x];
                counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] += 1;

                atomicExch(&mutex_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x], 0);
                blocked = false;
            }
        }
    }
    __syncthreads();
    //дальше пусть каждая стартовая нить блока обработает массив и найдем максимум
    if(threadIdx.x == 0 && threadIdx.y == 0 && counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x] != 0){
        int max = im_values[0];
        int index = 0;
        for(int i = 1; i != counter_for_each_block[blockIdx.y * GRID_SIZE_X + blockIdx.x]; i++){
            if (im_values[i] > max){
                max = im_values[i];
                index = i;
            }
        }
        //запись
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mLock, 0, 1)) {
                if(*global_counter <= VECTOR_INIT_CAPACITY){
                    vectorX[*global_counter] = block_vectorX[index];
                    vectorY[*global_counter] = block_vectorY[index];
                    *global_counter+=1;
                }
                atomicExch(&mLock, 0);
                blocked = false;
            }
        }
    }
    __syncthreads();
}


__global__ void DispRingFilterGPU(unsigned char* ImageData, int* vectorObjectX, int* vectorObjectY,
                                  int* counterObject, int* vectorResultX, int* vectorResultY, int* counterFiltered,
                                  int widthOfWindow, float SKO_Porog, int cols, int rows){
    if( (threadIdx.x + threadIdx.y*THREAD_DIM) >= *counterObject){
        return;
    }

    __shared__ int mutex;

    if(threadIdx.x == 0 && threadIdx.y == 0){
        mutex = 0;
    }

    int pointX = vectorObjectX[threadIdx.x + threadIdx.y * THREAD_DIM];
    int pointY = vectorObjectY[threadIdx.x + threadIdx.y * THREAD_DIM];

    if(pointX < widthOfWindow/2 || cols - pointX < widthOfWindow/2 ||
       pointY < widthOfWindow/2 || rows - pointY < widthOfWindow/2){
        return;
    }

    int M = 0;
    for(int l = -widthOfWindow/2+1; l != widthOfWindow/2; l++){
        M += (int)ImageData[(pointY - widthOfWindow/2) * cols + pointX + l];
        M += (int)ImageData[(pointY + widthOfWindow/2) * cols + pointX + l];
    }

    for(int l = -widthOfWindow/2+1; l != widthOfWindow/2; l++){
        M += (int)ImageData[(pointY + l) * cols + pointX - widthOfWindow/2];
        M += (int)ImageData[(pointY + l) * cols + pointX + widthOfWindow/2];
    }

    M = M/( (2*widthOfWindow + 2) + (2*widthOfWindow - 2));

    float SumForSKORing = 0;
    for(int l = -widthOfWindow/2; l != widthOfWindow/2 + 1; l++){
        SumForSKORing += ((int)ImageData[(pointY - widthOfWindow/2) * cols + pointX + l] - M) * ((int)ImageData[(pointY - widthOfWindow/2) * cols + pointX + l] - M);
        SumForSKORing += ((int)ImageData[(pointY + widthOfWindow/2) * cols + pointX + l] - M) * ((int)ImageData[(pointY + widthOfWindow/2) * cols + pointX + l] - M);
    }

    for(int l = -widthOfWindow/2 + 1; l != widthOfWindow/2; l++){
        SumForSKORing += ((int)ImageData[(pointY + l) * cols + pointX - widthOfWindow/2] - M) * ((int)ImageData[(pointY + l) * cols + pointX - widthOfWindow/2] - M);
        SumForSKORing += ((int)ImageData[(pointY + l) * cols + pointX + widthOfWindow/2] - M) * ((int)ImageData[(pointY + l) * cols + pointX + widthOfWindow/2] - M);
    }

    SumForSKORing = SumForSKORing/( (2*widthOfWindow + 2) + (2*widthOfWindow - 2));
    SumForSKORing = sqrt(SumForSKORing);

    if((int)ImageData[pointY * cols + pointX] >= (M + SKO_Porog * SumForSKORing) ){
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&mutex, 0, 1)) {
                vectorResultX[*counterFiltered] = pointX;
                vectorResultY[*counterFiltered] = pointY;
                *counterFiltered+=1;

                atomicExch(&mutex, 0);
                blocked = false;
            }
        }
    }

    __syncthreads();
}


extern "C" ContForPoints* GPUCalc(unsigned char* ImageData, int rows, int cols, int pud, int threshold,
                                  int widthForFilter, float SKO_Porog){
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
    int* dev_CounterGlobal;
    int* dev_CounterForEachBlock;
    int* dev_MutexForEachBlock;
    int* dev_GlobalMutex;

    int state = 0;

    cudaMalloc((void**)&dev_Image, sizeof(unsigned char) * cols * rows);
    cudaMalloc((void**)&dev_X, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_CounterGlobal, sizeof(int));
    cudaMalloc((void**)&dev_CounterForEachBlock, sizeof(int) * GRID_SIZE_Y * GRID_SIZE_X);
    cudaMalloc((void**)&dev_MutexForEachBlock, sizeof(int) * GRID_SIZE_Y * GRID_SIZE_X);

    cudaMemcpy(dev_Image, ImageData, cols*rows * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_CounterGlobal, &state, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_GlobalMutex, &state, sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM)); // x = 20 y = 16
    dim3 blockSize (THREAD_DIM, THREAD_DIM);

    RingFilterGPU<<<gridSize, blockSize>>>(
                dev_Image,
                dev_X,
                dev_Y,
                dev_CounterForEachBlock,
                dev_MutexForEachBlock,
                dev_CounterGlobal,
                rows,
                cols,
                pud,
                threshold
            );
    cudaDeviceSynchronize();

//    int* res_X = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
//    int* res_Y = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
//    int* res_counter = (int*)malloc(sizeof(int));
//    cudaError_t error1 = cudaMemcpy(res_X, dev_X, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
//    cudaError_t error2 = cudaMemcpy(res_Y, dev_Y, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
//    cudaError_t error3 = cudaMemcpy(res_counter, dev_CounterGlobal, sizeof(int), cudaMemcpyDeviceToHost);

    //filterDisp part
    int* dev_X_filtered;
    int* dev_Y_filtered;
    int* dev_CounterFiltered;
    cudaMalloc((void**)&dev_X_filtered, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_Y_filtered, sizeof(int) * VECTOR_INIT_CAPACITY);
    cudaMalloc((void**)&dev_CounterFiltered, sizeof(int));

    cudaMemcpy(dev_CounterFiltered, &state, sizeof(int), cudaMemcpyHostToDevice);

    DispRingFilterGPU<<<dim3(1,1), blockSize>>>(
                      dev_Image,
                      dev_X,
                      dev_Y,
                      dev_CounterGlobal,
                      dev_X_filtered,
                      dev_Y_filtered,
                      dev_CounterFiltered,
                      widthForFilter,
                      SKO_Porog,
                      cols,
                      rows
                  );
    cudaDeviceSynchronize();

    int* res_X = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* res_Y = (int*)malloc(sizeof(int) * VECTOR_INIT_CAPACITY);
    int* res_counter = (int*)malloc(sizeof(int));
    cudaError_t error1 = cudaMemcpy(res_X, dev_X_filtered, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error2 = cudaMemcpy(res_Y, dev_Y_filtered, sizeof(int) * VECTOR_INIT_CAPACITY, cudaMemcpyDeviceToHost);
    cudaError_t error3 = cudaMemcpy(res_counter, dev_CounterFiltered, sizeof(int), cudaMemcpyDeviceToHost);

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
    cudaFree(dev_CounterGlobal);
    cudaFree(dev_CounterForEachBlock);
    cudaFree(dev_MutexForEachBlock);

    cudaFree(dev_X_filtered);
    cudaFree(dev_Y_filtered);
    cudaFree(dev_CounterFiltered);

    return cont;
}
