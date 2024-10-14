#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#define THREAD_DIM 32//размер 1024, если брать квадратный блок то 32 на 32 нитей в блоке
#define NUM_STREAMS 32

#define VECTOR_INIT_CAPACITY 5000

typedef struct ContForPoints_
{
    int* vectorX;
    int* vectorY;
    int* counter;
} ContForPoints;

extern "C" ContForPoints* GPUCalc(unsigned char*, int, int, int, int);

#endif // CUDA_HEADER_H
