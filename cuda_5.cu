#include "common.hpp"

// 双缓冲/预取，原理是指令级并行，如果指令之间相互独立，没有读写依赖，
// 线程读写操作的顺序和指令在代码中出现的顺序不一定相同
// __syncthreads函数可以确保该线程块在障碍点之前的读写操作已经完成
// __threadfence_block函数不进行障碍同步，只是挂起当前线程，直到之前的写入操作刷新完毕
// __threadfence_block/__threadfence/__threadfence_system对应刷新的内存和缓存层次不断提升

// block_shape应能整除M、K、N，block_unit应能整除K
constexpr size_t block_shape = 128, block_unit = 8;
// thread_shape应能整除block_shape和block_unit
constexpr size_t thread_shape = 8, block_dim = block_shape / thread_shape;
// thread_unit应能整除block_unit
constexpr size_t thread_unit = 1;

__device__ void kernel_thread(const real (*A)[block_unit], const real (*B)[block_shape], real (*C)[thread_shape])
{
    unsigned ty = threadIdx.y, tx = threadIdx.x;

    // 不是协同拷贝，每个线程拷贝自己所需的数据，不同线程会重复拷贝数据
    real r_a[thread_shape][thread_unit], r_b[thread_unit][thread_shape];

    for (size_t i = 0; i < block_unit / thread_unit; ++i) {
        for (size_t j = 0; j < thread_shape; ++j) {
            for (size_t k = 0; k < thread_unit; ++k) {
                r_a[j][k] = A[ty + j * block_dim][k + i * thread_unit];
            }
        }
        for (size_t k = 0; k < thread_unit; ++k) {
            for (size_t j = 0; j < thread_shape; ++j) {
                r_b[k][j] = B[k + i * thread_unit][tx + j * block_dim];
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

    __shared__ real s_a[2][block_shape][block_unit], s_b[2][block_unit][block_shape];

    real sum[thread_shape][thread_shape];
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; ++q) {
            sum[p][q] = 0.0;
        }
    }

    // 取第一部分
    unsigned smem_stage_idx = 0;
    for (size_t j = 0; j < thread_shape; ++j) {
        for (size_t k = 0; tx + k * block_dim < block_unit; ++k) {
            // 安培之前的架构，从全局内存转移到共享内存会经过寄存器中转
            s_a[smem_stage_idx][ty + j * block_dim][tx + k * block_dim] =
                A[iy + j * block_dim][tx + k * block_dim];
        }
    }
    for (size_t k = 0; ty + k * block_dim < block_unit; ++k) {
        for (size_t j = 0; j < thread_shape; ++j) {
            s_b[smem_stage_idx][ty + k * block_dim][tx + j * block_dim] =
                B[ty + k * block_dim][ix + j * block_dim];
        }
    }
    // 协同拷贝，等待拷贝结束
    __syncthreads();

    // 调整循环下标
    for (size_t i = 1; i <= K / block_unit; ++i) {
        if (i != K / block_unit) {
            // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
            size_t col_a = i * block_unit + tx, row_b = i * block_unit + ty;
            for (size_t j = 0; j < thread_shape; ++j) {
                for (size_t k = 0; tx + k * block_dim < block_unit; ++k) {
                    // 安培之前的架构，从全局内存转移到共享内存会经过寄存器中转
                    s_a[smem_stage_idx ^ 1][ty + j * block_dim][tx + k * block_dim] =
                        A[iy + j * block_dim][col_a + k * block_dim];
                }
            }
            for (size_t k = 0; ty + k * block_dim < block_unit; ++k) {
                for (size_t j = 0; j < thread_shape; ++j) {
                    s_b[smem_stage_idx ^ 1][ty + k * block_dim][tx + j * block_dim] =
                        B[row_b + k * block_dim][ix + j * block_dim];
                }
            }
        }

        kernel_thread(s_a[smem_stage_idx], s_b[smem_stage_idx], sum);

        if (i != K / block_unit) {
            // 避免在共享内存使用之前被修改
            __syncthreads();
        }
        // 切换目标缓冲区
        smem_stage_idx ^= 1;
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