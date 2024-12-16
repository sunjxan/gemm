#include "common.hpp"

void matmul(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    real *B2;
    CHECK(cudaMallocHost(&B2, KN_size));
    real (*nB2)[K] = reinterpret_cast<decltype(nB2)>(B2);

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            nB2[j][i] = nB[i][j];
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            real sum = 0.0;
            for (size_t t = 0; t < K; ++t) {
                sum += nA[i][t] * nB2[j][t];
            }
            nC[i][j] = sum;
        }
    }

    CHECK(cudaFreeHost(B2));
}

int main()
{
    launch_cpu();
    return 0;
}