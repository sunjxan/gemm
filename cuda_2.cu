#include <cstdio>

#include "error.cuh"
#include "common.cuh"

// 1. 为了避免竞争条件，一个线程完成一个或多个坐标点的矩阵乘法，在K轴上不分多个线程
// 2. 使用共享内存，为达到最高使用效率，一个线程块应该完成一个正方形范围坐标的矩阵乘法,要做边角处理
// 3. 共享内存容量有限，从全局内存加载到共享内存的过程，要在K轴上分段进行

constexpr size_t blockSize = 32, unit = 16, reg_size = DIVUP(unit, blockSize);

__global__ void kernel(const real *A, const real *B, real *C, size_t unit_size)
{
    // s_a前半部分是blockDim.y行unit列
    extern __shared__ real s_a[];
    // s_a后半部分是unit行blockDim.x列
    real *s_b = s_a + unit_size;
    unsigned ty = threadIdx.y, bdy = blockDim.y, iy = blockIdx.y * bdy + ty;
    unsigned tx = threadIdx.x, bdx = blockDim.x, ix = blockIdx.x * bdx + tx;
    real sum = 0.0f;
    // 安培之前的架构，从全局内存转移到共享内存需要经过寄存器，并做块同步
    real a[reg_size], b[reg_size];

    for (size_t i = 0; i < DIVUP(K, unit); ++i) {
        // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
        size_t iu = i * unit, col_a = iu + tx, row_b = iu + ty;
        // 边角处理
        if (iy < M && col_a < K) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + tx < unit) {
                    a[j] = A[iy * K + (beg + col_a)];
                }
                beg += bdx;
            }
        }
        // 边角处理
        if (row_b < K && ix < N) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + ty < unit) {
                    b[j] = B[(beg + row_b) * N + ix];
                }
                beg += bdy;
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
                beg += bdx;
            }
        }
        // 边角处理
        if (row_b < K && ix < N) {
            size_t beg = 0;
            for (size_t j = 0; j < reg_size; ++j) {
                if (beg + ty < unit) {
                    s_b[(beg + ty) * bdx + tx] = b[j];
                }
                beg += bdy;
            }
        }
        __syncthreads();
        // 边角处理
        if (iy < M && ix < N) {
            for (size_t j = 0; j < unit; ++j) {
                // 边角处理
                if (iu + j >= K) {
                    break;
                }
                sum += s_a[ty * unit + j] * s_b[j * bdx + tx];
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
    dim3 block_size(blockSize, blockSize);
    dim3 grid_size(DIVUP(M, block_size.x), DIVUP(N, block_size.y));
    size_t unit_size = unit * blockSize, shared_size = (unit_size * real_size) << 1;
    kernel<<<grid_size, block_size, shared_size>>>(d_A, d_B, d_C, unit_size);
    CHECK(cudaGetLastError());
}

int main()
{
    launch_gpu();
    return 0;
}
