#include <cstdio>

#include "error.cuh"
#include "common.cuh"

// 1. 为了避免竞争条件，一个线程完成一个或多个坐标点的矩阵乘法，在K轴上不分多个线程
// 2. 使用共享内存，为达到最高使用效率，一个线程块应该完成一个正方形范围坐标的矩阵乘法
// 3. 共享内存容量有限，从全局内存加载到共享内存的过程，要在K轴上分段进行

__global__ void kernel(const real *A, const real *B, real *C, size_t unit, size_t unit_size)
{
    extern __shared__ real s_a[];
    real *s_b = s_a + unit_size;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y, tx = threadIdx.x;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x, ty = threadIdx.y;
    size_t pos = iy * N + ix;

    for (size_t i = 0; i < DIVUP(K, unit); ++i) {
        size_t col_a = tx + i * unit, row_b = ty + i * unit;
        real a = 0, b = 0;
        if (iy < M && col_a < K) {
            a = A[iy * K + col_a];
        }
        if (row_b < K && ix < N) {
            b = B[row_b * N + ix];
        }
        __syncthreads();
        if (iy < M && col_a < K) {
            s_a[ty * unit + tx] = a;
        }
        if (row_b < K && ix < N) {
            s_b[ty * unit + tx] = b;
        }
        __syncthreads();
        if (iy < M && ix < N) {
            for (size_t j = 0; j < unit; ++j) {
                if (i * unit + j >= K) {
                    break;
                }
                C[pos] += s_a[ty * unit + j] * s_b[j * unit + tx];
            }
        }
    }
}

void gemm(const real *d_A, const real *d_B, real *d_C)
{
    CHECK(cudaMemset(d_C, 0, MN_size));

    unsigned length = 32;
    dim3 block_size(length, length);
    dim3 grid_size(DIVUP(M, length), DIVUP(N, length));
    // unit=length，K轴分段是个正方形
    size_t unit = length, unit_size = unit * length, shared_size = (unit_size * real_size) << 1;
    kernel<<<grid_size, block_size, shared_size>>>(d_A, d_B, d_C, unit, unit_size);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
