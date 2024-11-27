#include <cstdio>

#include "error.cuh"
#include "common.cuh"

__global__ void kernel(const real *A, const real *B, real *C)
{
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy < M && ix < N) {
        real sum = 0;
        for (size_t t = 0; t < K; ++t) {
            sum += A[iy * K + t] * B[t * N + ix];
        }
        C[iy * N + ix] = sum;
    }
}

void gemm(const real *d_A, const real *d_B, real *d_C)
{
    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(M, block_size.x), DIVUP(N, block_size.y));
    kernel<<<grid_size, block_size>>>(d_A, d_B, d_C);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
