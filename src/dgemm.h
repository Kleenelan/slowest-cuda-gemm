#pragma once
#include "cuda_runtime.h"
typedef long int lint;

extern "C"
void blas_dgemm(cudaStream_t    stream,
                int             opA,
                int             opB,
                lint            M,
                lint            N,
                lint            K,
                double*         A,
                lint            lda,
                double*         B,
                lint            ldb,
                double*         C,
                lint            ldc,
                double          alpha,
                double          beta);
