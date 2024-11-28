#include <cstdio>

#include "common.cuh"

// 朴素实现，注意iy和ix对行列的编码

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[K])
{
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy < M && ix < N) {
        real sum = 0.0f;
        for (size_t t = 0; t < K; ++t) {
            sum += A[iy][t] * B[t][ix];
        }
        C[iy][ix] = sum;
    }
}

void gemm(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(M, block_size.x), DIVUP(N, block_size.y));
    kernel<<<grid_size, block_size>>>(nA, nB, nC);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
