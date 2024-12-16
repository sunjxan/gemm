#include <cublas_v2.h>

#include "common.hpp"

// 将A和B交换，因为1.AxB=C等价于转置Bx转置A=转置C，2.矩阵行优先存储和转置矩阵列优先存储结果一致，转置等价于改变存储layout
// 所以生成列优先矩阵C的任务可以转换为生成行优先矩阵C转置，仍使用生成行优先矩阵的操作过程，但A和B需要调换位置，且都要转置（改变存储layout）
// cublas默认列优先存储矩阵

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