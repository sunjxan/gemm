#include "common.hpp"

void matmul(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            real sum = 0.0;
            for (size_t t = 0; t < K; ++t) {
                sum += nA[i][t] * nB[t][j];
            }
            nC[i][j] = sum;
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}