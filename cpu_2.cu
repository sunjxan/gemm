#include <cstdio>

#include "common.cuh"

void gemm(const real *h_A, const real *h_B, real *h_C)
{
    for (size_t t = 0; t < K; ++t) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                h_C[i * N + j] += h_A[i * K + t] * h_B[t * N + j];
            }
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}
