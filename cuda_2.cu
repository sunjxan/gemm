#include <cstdio>

#include "common.hpp"

// 1. 为了避免竞争条件，一个线程完成一个或多个坐标点的矩阵乘法，在K轴上不分多个线程
// 2. 使用共享内存缓存线程块负责的子矩阵的计算数据，为达到最高使用效率，子矩阵应是正方形
// 3. 共享内存容量有限，从全局内存加载到共享内存计算的过程，应在K轴上分段进行

// block_shape应能整除M、K、N，unit应能整除K
// unit取简单情况等于block_shape，一次取满
constexpr size_t block_shape = 32, unit = block_shape;

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{
    unsigned ty = threadIdx.y, iy = blockIdx.y * block_shape + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_shape + tx;

    __shared__ real s_a[block_shape][unit], s_b[unit][block_shape];

    real sum = 0.0f;
    for (size_t i = 0; i < K / unit; ++i) {
        // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
        size_t col_a = i * unit + tx, row_b = i * unit + ty;
        // 安培之前的架构，从全局内存转移到共享内存会经过寄存器中转
        s_a[ty][tx] = A[iy][col_a];
        s_b[ty][tx] = B[row_b][ix];
        // 协同拷贝，等待拷贝结束
        __syncthreads();
        for (size_t j = 0; j < unit; ++j) {
            sum += s_a[ty][j] * s_b[j][tx];
        }
        // 避免在共享内存使用之前被修改
        if (i != K / unit - 1) {
            __syncthreads();
        }
    }
    C[iy][ix] = sum;
}

void gemm(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    // 线程块是正方形
    dim3 block_size(block_shape, block_shape);
    // N是列对应x，M是行对应y
    dim3 grid_size(N / block_shape, M / block_shape);
    kernel<<<grid_size, block_size>>>(nA, nB, nC);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
