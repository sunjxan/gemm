#include "common.hpp"

void matmul(const real *A, const real *B, real *C)
{
    CHECK(cudaMemset(C, 0, MN_size));

    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    size_t Mtile = 128, Ntile = 128;
    for (size_t m = 0; m < M; m += Mtile) {
        for (size_t n = 0; n < N; n += Ntile) {
            for (size_t t = 0; t < K; ++t) {
                for (size_t i = 0; i < Mtile; ++i) {
                    for (size_t j = 0; j < Ntile; ++j) {
                        size_t row = m + i, col = n + j;
                        nC[row][col] += nA[row][t] * nB[t][col];
                    }
                }
            }
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}
