#include <cstdio>

#include "common.cuh"

void gemm(const real *A, const real *B, real *C)
{
    CHECK(cudaMemset(C, 0, MN_size));

    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    for (size_t t = 0; t < K; ++t) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                nC[i][j] += nA[i][t] * nB[t][j];
            }
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}
