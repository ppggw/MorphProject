#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H


#define THREAD_DIM 32//размер 1024, если брать квадратный блок то 32 на 32 нитей в блоке
#define NUM_STREAMS 32

typedef struct FakeMat_ {
    unsigned char *Ptr;
    int rows;
    int cols;
} FakeMat;

extern "C" unsigned char* GPUCalc(unsigned char*, int, int, int, int);

#endif // CUDA_HEADER_H
