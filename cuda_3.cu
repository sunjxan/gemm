#include "common.hpp"

// 1个线程完成 n x n 个线程的工作

// block_shape应能整除M、K、N，block_unit应能整除K
constexpr size_t block_shape = 128, block_unit = 8;
// thread_shape应能整除block_shape和block_unit
constexpr size_t thread_shape = 8, block_dim = block_shape / thread_shape;

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{
    unsigned ty = threadIdx.y, iy = blockIdx.y * block_shape + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_shape + tx;

    __shared__ real s_a[block_shape][block_unit], s_b[block_unit][block_shape];

    real sum[thread_shape][thread_shape];
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; ++q) {
            sum[p][q] = 0.0;
        }
    }
    for (size_t i = 0; i < K / block_unit; ++i) {
        // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
        size_t col_a = i * block_unit + tx, row_b = i * block_unit + ty;
        for (size_t j = 0; j < thread_shape; ++j) {
            for (size_t k = 0; tx + k * block_dim < block_unit; ++k) {
                // 安培之前的架构，从全局内存转移到共享内存会经过寄存器中转
                s_a[ty + j * block_dim][tx + k * block_dim] =
                    A[iy + j * block_dim][col_a + k * block_dim];
            }
        }
        for (size_t k = 0; ty + k * block_dim < block_unit; ++k) {
            for (size_t j = 0; j < thread_shape; ++j) {
                s_b[ty + k * block_dim][tx + j * block_dim] =
                    B[row_b + k * block_dim][ix + j * block_dim];
            }
        }
        // 协同拷贝，等待拷贝结束
        __syncthreads();
        for (size_t j = 0; j < block_unit; ++j) {
            for (size_t p = 0; p < thread_shape; ++p) {
                for (size_t q = 0; q < thread_shape; ++q) {
                    sum[p][q] += s_a[ty + p * block_dim][j] * s_b[j][tx + q * block_dim];
                }
            }
        }
        if (i != K / block_unit - 1) {
            // 避免在共享内存使用之前被修改
            __syncthreads();
        }
    }
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; ++q) {
            // 注意sum计算和传值的对应方式
            C[iy + p * block_dim][ix + q * block_dim] = sum[p][q];
        }
    }
}

void matmul(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    // 线程块是正方形
    dim3 block_size(block_dim, block_dim);
    // N是列对应x，M是行对应y
    dim3 grid_size(N / block_shape, M / block_shape);
    kernel<<<grid_size, block_size>>>(nA, nB, nC);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}