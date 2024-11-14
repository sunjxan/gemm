#pragma once
#include <cstdio>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = (call);            \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#define CHECK_CUBLAS(call)                            \
do                                                    \
{                                                     \
    const cublasStatus_t status = (call);             \
    if (status != CUBLAS_STATUS_SUCCESS)              \
    {                                                 \
        printf("CUBLAS Error:\n");                    \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", status);       \
        exit(1);                                      \
    }                                                 \
} while (0)

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1e-12;
#else
    typedef float real;
    const real EPSILON = 5e-3f;
#endif
