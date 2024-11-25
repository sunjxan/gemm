#include <cstdio>
#include <cublas_v2.h>

#include "common.cuh"

void gemm(const real *d_A, const real *d_B, real *d_C)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    real alpha = 1.0f, beta = 0.0f;
    #ifdef USE_DP
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_B, N, d_A, K, &beta, d_C, N));
    #else
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_B, N, d_A, K, &beta, d_C, N));
    #endif
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main()
{
    launch_gpu();
    return 0;
}
