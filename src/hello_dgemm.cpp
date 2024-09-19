#include <stdio.h>
#include "dgemm.hpp"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "init_matrix.hpp"
/*
void cucublasDgemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n, int k,
                    const double *alpha,
                    const double *A, int lda,
                    const double *B, int ldb,
                    const double *beta,
                    double *C, int ldc);*/


extern "C"
void dgemm_(char *, char *, lint *, lint *, lint *, double *,
            double *, lint *, double *, lint *, double *, double *, lint *);

//typedef  int lint;
typedef double dbl;

void print_matrix_d(lint M, lint N, dbl* A, lint lda)
{
    for(int i=M-12; i<M; i++){
        for(int j=N-12; j<N; j++){
            printf("%8.3f ", A[i + j*lda]);
        }
        printf("\n");
    }
}

void cpu_dgemm_nn(  int opA,
                    int opB,
                    lint M,
                    lint N,
                    lint K,
                    double alpha,
                    double* A,
                    lint lda,
                    double* B,
                    lint ldb,
                    double beta,
                    double* C,
                    lint ldc)
{
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            double tmp = 0.0;
            for(int k=0; k<K; k++){
                tmp += A[i + k*lda]*B[k + j*ldb];
            }
            C[i + j*ldc] = alpha*tmp + beta*C[i + j*ldc];
        }
    }
}


void cpu_dgemm_nt(  int opA,
                    int opB,
                    lint M,
                    lint N,
                    lint K,
                    double alpha,
                    double* A,
                    lint lda,
                    double* B,
                    lint ldb,
                    double beta,
                    double* C,
                    lint ldc)
{
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            double tmp = 0.0;
            for(int k=0; k<K; k++){
                tmp += A[i + k*lda]*B[k*ldb + j];
            }
            C[i + j*ldc] = alpha*tmp + beta*C[i + j*ldc];
        }
    }
}

int main()
{
    lint M = 33330;//1024;
    lint N = 33340;//2048;
    lint K = 384;//512;

    lint lda = M;
    lint ldb = N;
    lint ldc = N;

    dbl *A = nullptr;
    dbl *B = nullptr;
    dbl *C = nullptr;

    dbl *C_h = nullptr;

    A = (dbl*)malloc(lda*K*sizeof(dbl));
    B = (dbl*)malloc(ldb*K*sizeof(dbl));
    C = (dbl*)malloc(ldc*N*sizeof(dbl));
    C_h = (dbl*)malloc(ldc*N*sizeof(dbl));


    dbl *A_d = nullptr;
    dbl *B_d = nullptr;
    dbl *C_d = nullptr;

    cudaMalloc((void**)&A_d, lda*K*sizeof(double));
    cudaMalloc((void**)&B_d, ldb*K*sizeof(double));
    cudaMalloc((void**)&C_d, ldc*N*sizeof(double));


    //void srand_rand_double(int seed, double* A, unsigned long int len);
    srand_rand_double(2024, A, lda*K);
    cudaMemcpy(A_d, A, lda*K*sizeof(double), cudaMemcpyHostToDevice);

    printf("A =\n");
    print_matrix_d(M, K, A, lda);

    srand_rand_double(2025, B, ldb*K);
    cudaMemcpy(B_d, B, ldb*K*sizeof(double), cudaMemcpyHostToDevice);

    printf("B =\n");
    print_matrix_d(N, K, B, ldb);

    srand_rand_double(2026, C, ldc*N);
    cudaMemcpy(C_d, C, ldc*N*sizeof(double), cudaMemcpyHostToDevice);

    printf("C =\n");
    print_matrix_d(M, N, C, ldc);

    srand_rand_double(2026, C_h, ldc*N);
    printf("C_h =\n");
    print_matrix_d(M, N, C_h, ldc);

//void dgemm_(char *, char *, lint *, lint *, lint *, double *,
//           double *, lint *, double *, lint *, double *, double *, int *);

    double alpha = 0.001;
    double beta = 0.002;

    dgemm_("N", "T", &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    printf("C(a*A*B + b*C cpu) =\n");
    print_matrix_d(M, N, C, ldc);

#if 0
    cpu_dgemm_nt(1, 1, M, N, K, alpha, A, lda, B, ldb, beta, C_h, ldc);
    printf("C_h =\n");
    print_matrix_d(M, N, C_h, ldc);
#endif
    memset(C_h, 0x00, ldc*N*sizeof(dbl));
    //sleep(2);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);


    blas_dgemm_slowest(stream, 1, 1, M, N, K, alpha, A_d, lda, B_d, ldb, beta, C_d, ldc);
    cudaDeviceSynchronize();
    cudaMemcpy(C_h, C_d, ldc*N*sizeof(dbl), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("C_h(a*A*B + b*C cpu gpu ) =\n");
    print_matrix_d(M, N, C_h, ldc);


    return 0;
}



