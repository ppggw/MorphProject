#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#define THREAD_DIM 32//размер 1024, если брать квадратный блок то 32 на 32 нитей в блоке
#define GRID_SIZE_X 20
#define GRID_SIZE_Y 16
#define VECTOR_INIT_CAPACITY 600

typedef struct ContForPoints_
{
    int* vectorX;
    int* vectorY;
    int* counter;
} ContForPoints;

extern "C" ContForPoints* GPUCalc(unsigned char*, int, int, int, int, int, float);

#endif // CUDA_HEADER_H
