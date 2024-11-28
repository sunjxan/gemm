#include <cstdio>

#include "common.hpp"

// 一个线程完成4X4个线程的工作

// block_shape应能整除M、K、N，block_unit应能整除K
constexpr size_t block_shape = 32, block_unit = 16;
// thread_shape应能整除block_shape和block_unit
constexpr size_t thread_shape = 4, block_dim = block_shape / thread_shape;
// thread_unit应能整除block_unit
constexpr size_t thread_unit = 2, frag_size = block_unit / block_dim;

__device__ void kernel_thread(const real (*A)[block_unit], const real (*B)[block_shape], real (*C)[thread_shape])
{
    unsigned ty = threadIdx.y, tx = threadIdx.x;

    // 不是协同拷贝，每个线程拷贝自己所需的数据，不同线程会重复拷贝数据
    real r_a[thread_shape][thread_unit], r_b[thread_unit][thread_shape];

    for (size_t i = 0; i < block_unit / thread_unit; ++i) {
        for (size_t j = 0; j < thread_shape; ++j) {
            for (size_t k = 0; k < thread_unit; ++k) {
                r_a[j][k] = A[ty + j * block_dim][i * thread_unit + k];
            }
        }
        for (size_t k = 0; k < thread_unit; ++k) {
            for (size_t j = 0; j < thread_shape; ++j) {
                r_b[k][j] = B[i * thread_unit + k][tx + j * block_dim];
            }
        }
        for (size_t j = 0; j < thread_unit; ++j) {
            for (size_t p = 0; p < thread_shape; ++p) {
                for (size_t q = 0; q < thread_shape; ++q) {
                    C[p][q] += r_a[p][j] * r_b[j][q];
                }
            }
        }
    }
}

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{

    unsigned ty = threadIdx.y, iy = blockIdx.y * block_shape + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_shape + tx;

    __shared__ real s_a[block_shape][block_unit], s_b[block_unit][block_shape];

    real sum[thread_shape][thread_shape], frag_a[thread_shape][frag_size], frag_b[frag_size][thread_shape];
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; ++q) {
            sum[p][q] = 0.0f;
        }
    }
    // 安培之前的架构，从全局内存转移到共享内存需要经过寄存器，并做块同步
    for (size_t i = 0; i < K / block_unit; ++i) {
        // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
        size_t i_block_unit = i * block_unit, col_a = i_block_unit + tx, row_b = i_block_unit + ty;
        for (size_t j = 0; j < thread_shape; ++j) {
            for (size_t k = 0; k < frag_size; ++k) {
                frag_a[j][k] = A[iy + j * block_dim][col_a + k * block_dim];
            }
        }
        for (size_t k = 0; k < frag_size; ++k) {
            for (size_t j = 0; j < thread_shape; ++j) {
                frag_b[k][j] = B[row_b + k * block_dim][ix + j * block_dim];
            }
        }
        __syncthreads();
        for (size_t j = 0; j < thread_shape; ++j) {
            for (size_t k = 0; k < frag_size; ++k) {
                s_a[ty + j * block_dim][tx + k * block_dim] = frag_a[j][k];
            }
        }
        for (size_t k = 0; k < frag_size; ++k) {
            for (size_t j = 0; j < thread_shape; ++j) {
                s_b[ty + k * block_dim][tx + j * block_dim] = frag_b[k][j];
            }
        }
        __syncthreads();
        kernel_thread(s_a, s_b, sum);
    }
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; ++q) {
            // 注意sum计算和传值的对应方式
            C[iy + p * block_dim][ix + q * block_dim] = sum[p][q];
        }
    }
}

void gemm(const real *A, const real *B, real *C)
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
}

int main()
{
    launch_gpu();
    return 0;
}
