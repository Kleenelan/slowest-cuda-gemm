#include "dgemm.h"
#include "dgemm.hpp"
typedef long int lint;


void blas_dgemm_slowest
                (cudaStream_t   stream,
                int             opA,
                int             opB,
                lint            M,
                lint            N,
                lint            K,
                double          alpha,
                double*         A,
                lint            lda,
                double*         B,
                lint            ldb,
                double          beta,
                double*         C,
                lint            ldc)
{

    blas_dgemm( stream,
                opA,
                opB,
                M,
                N,
                K,
                A,
                lda,
                B,
                ldb,
                C,
                ldc,
                alpha,
                beta);

}