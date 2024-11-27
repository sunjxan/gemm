#include <cstdio>

#include "error.cuh"
#include "common.cuh"

// 1. 为了避免竞争条件，一个线程完成一个或多个坐标点的矩阵乘法，在K轴上不分多个线程
// 2. 使用共享内存，为达到最高使用效率，一个线程块应该完成一个正方形范围坐标的矩阵乘法,要做边角处理
// 3. 共享内存容量有限，从全局内存加载到共享内存的过程，要在K轴上分段进行

constexpr size_t block_shape = 32, unit = 16;
constexpr size_t thread_shape = 4, block_dim = DIVUP(block_shape, thread_shape);
constexpr size_t reg_size = DIVUP(unit, block_dim);

__global__ void kernel(const real *A, const real *B, real *C, size_t unit_size)
{
    // s_a前半部分是block_shape行unit列
    extern __shared__ real s_a[];
    // s_a后半部分是unit行block_shape列
    real *s_b = s_a + unit_size;
    unsigned ty = threadIdx.y, by = blockIdx.y * block_shape, iy = blockIdx.y * block_dim + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_dim + tx;
    real sum = 0.0f;
    // 安培之前的架构，从全局内存转移到共享内存需要经过寄存器，并做块同步
    real a[reg_size][thread_shape], b[reg_size][thread_shape];

    for (size_t i = 0; i < DIVUP(K, unit); ++i) {
        // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
        size_t i_unit = i * unit, col_a = i_unit + tx, row_b = i_unit + ty;
        // 边角处理 重构
        if (iy < M && col_a < K) {
            for (size_t t = 0; t < thread_shape; ++t) {
                size_t beg_x = 0, beg_y = by;
                for (size_t j = 0; j < reg_size; ++j) {
                    if (beg_x + tx < unit) {
                        a[j][t] = A[(beg_y + t) * K + (beg_x + col_a)];
                    }
                    beg_x += block_dim;
                    beg_y += thread_shape;
                }
            }
        }
        // 边角处理
        if (row_b < K && ix < N) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + ty < unit) {
                    b[j] = B[(beg + row_b) * N + ix];
                }
                beg += block_dim;
            }
        }
        __syncthreads();
        // 边角处理
        if (iy < M && col_a < K) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + tx < unit) {
                    s_a[ty * unit + (beg + tx)] = a[j];
                }
                beg += block_dim;
            }
        }
        // 边角处理
        if (row_b < K && ix < N) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + ty < unit) {
                    s_b[(beg + ty) * block_shape + tx] = b[j];
                }
                beg += block_dim;
            }
        }
        __syncthreads();
        // 边角处理
        if (iy < M && ix < N) {
            for (size_t j = 0; j < unit; ++j) {
                // 边角处理
                if (i_unit + j >= K) {
                    break;
                }
                sum += s_a[ty * unit + j] * s_b[j * block_shape + tx];
            }
        }
    }

    if (iy < M && ix < N) {
        C[iy * N + ix] = sum;
    }
}

void gemm(const real *d_A, const real *d_B, real *d_C)
{
    // 线程块是正方形
    dim3 block_size(block_dim, block_dim);
    dim3 grid_size(DIVUP(M, block_shape), DIVUP(N, block_shape));
    size_t unit_size = unit * block_shape, shared_size = (unit_size * real_size) << 1;
    kernel<<<grid_size, block_size, shared_size>>>(d_A, d_B, d_C, unit_size);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
