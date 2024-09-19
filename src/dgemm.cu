#include "cuda_runtime.h"
#include "init_matrix.hpp"

typedef long int lint;
typedef double dbl;

#define TM 1    //thread-block M
#define TN 1    //thread-block N
#define LK 1    //loop len of K

#define BX 32   //block.x
#define BY 32   //block.y

// C = a*A*B+ b*C
__global__
void dgemm_slowest01( int opA,
                    int opB,
                    lint M,
                    lint N,
                    lint K,
                    double* Ad,
                    lint lda,
                    double* Bd,
                    lint ldb,
                    double* Cd,
                    lint ldc,
                    double alpha,
                    double beta )
{
    lint BBi = (blockIdx.x * BX) * TM;    //Block begin i;
    lint BBj = (blockIdx.y * BY) * TN;    //Block begin j;
    lint ci = (threadIdx.x * TM) + BBi;   //C begin i; offset-i + BBi;
    lint cj = (threadIdx.y * TN) + BBj;   //C begin j; offset-j + BBj;
    if(threadIdx.x == 7 && threadIdx.y == 7)
        printf("T(%u, %u)ci = %ld, cj = %ld\n", threadIdx.x, threadIdx.y, ci, cj);

    double A[TM][LK];
    double B[TN][LK];
    double C[TM][TN];
    // fetch C_b = beta*C_b
    for(int bk=0; bk<K; bk+=LK)            //K loop, bk begin k of this K loop;
    {
        // fetch A_b
        #pragma unroll
        for(int ti=0; ti<TM; ti++){     //A row loop
            #pragma unroll
            for(int lk=0; (lk<LK)&&(bk+lk<K); lk++){  //A col loop
                lint i = ci + ti;
                lint k = bk + lk;

                if(i<M && bk+lk<K)
                    A[ti][lk] = Ad[i + k*lda];
                else
                    A[ti][lk] = 0.0;
            }
        }

        // fetch B_b
        #pragma unroll
        for(int tj=0; tj<TN; tj++){
            #pragma unroll
            for(int lk=0; (lk<LK)&&(bk+lk<K); lk++){
                lint j = cj + tj;
                lint k = bk + lk;
                if(j<N && bk +lk <K)
                    B[tj][lk] = Bd[k + j*ldb];
                else
                    B[tj][lk] = 0.0;
            }
        }

        // gemm(A,B,C)
        #pragma unroll
        for(int ti = 0; ti<TM; ti++){
            #pragma unroll
            for(int tj=0; tj<TN; tj++){
                #pragma unroll
                for(int lk=0; lk<LK; lk++){
                    C[ti][tj] += A[ti][lk] * B[tj][lk];

                }
            }
        }

        //store C
        #pragma unroll
        for(int ti = 0; ti<TM; ti++){
            #pragma unroll
            for(int tj=0; tj<TN; tj++){
                lint i = BBi + ti;
                lint j = BBj + tj;
                if(i<M && j<N)
                    Cd[i + j*ldc] = alpha*C[ti][tj] + beta*Cd[i + j*ldc];
            }
        }
    }
}


__global__
void dgemm_cuda_nt( int opA,
                    int opB,
                    lint M,
                    lint N,
                    lint K,
                    double* Ad,
                    lint lda,
                    double* Bd,
                    lint ldb,
                    double* Cd,
                    lint ldc,
                    double alpha,
                    double beta )
{
    lint i = blockIdx.x * blockDim.x + threadIdx.x;
    lint j = blockIdx.y * blockDim.y + threadIdx.y;
    double sigma = 0.0;
    if(i<M && j<N){
        for(lint k=0; k<K; k++)
            sigma += Ad[i + k*lda]*Bd[k*ldb + j];

        Cd[i + j*ldc] = alpha*sigma + beta*Cd[i + j*ldc];
    }
}

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
                double          beta)
{
    dim3 block_;
    dim3 grid_;

    block_.x = BX;
    block_.y = BY;
    //32*32*(4x4)=128x128
    //grid_.x = (M+BX*TM-1)/(BX*TM);
    //grid_.y = (N+BY*TN-1)/(BY*TN);

    grid_.x = (M+BX-1)/(BX);
    grid_.y = (N+BY-1)/(BY);

    printf("block_.x = %d, block_.y = %d, grid_.x = %d, grid_.y = %d\n", block_.x, block_.y, grid_.x, grid_.y);
    dgemm_cuda_nt<<<grid_, block_, 0, stream>>>(opA,
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
