#include <cublas_v2.h>

#include "common.hpp"

void matmul(const real *A, const real *B, real *C)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    real alpha = 1.0, beta = 0.0;
    #ifdef USE_DP
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, B, N, A, K, &beta, C, N));
    #else
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, B, N, A, K, &beta, C, N));
    #endif
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main()
{
    launch_gpu();
    return 0;
}
