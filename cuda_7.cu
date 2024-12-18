#include "common.hpp"

// 让最后一个小迭代为下一轮预取数据

// block_shape应能整除M、K、N，block_unit应能整除K
constexpr size_t block_shape = 128, block_unit = 8;
// thread_shape应能整除block_shape和block_unit
constexpr size_t thread_shape = 8, block_dim = block_shape / thread_shape;
// thread_unit应能整除block_unit
constexpr size_t thread_unit = 1;

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{

    unsigned ty = threadIdx.y, iy = blockIdx.y * block_shape + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_shape + tx;

    __shared__ real s_a[2][block_shape][block_unit], s_b[2][block_unit][block_shape];

    // 不是协同拷贝，每个线程拷贝自己所需的数据，不同线程会重复拷贝数据
    real r_a[2][thread_shape][thread_unit], r_b[2][thread_unit][thread_shape];

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

    // 取第一部分
    unsigned reg_stage_idx = 0;
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_unit; ++q) {
            r_a[reg_stage_idx][p][q] = s_a[smem_stage_idx][ty + p * block_dim][q];
        }
    }
    for (size_t q = 0; q < thread_unit; ++q) {
        for (size_t p = 0; p < thread_shape; ++p) {
            r_b[reg_stage_idx][q][p] = s_b[smem_stage_idx][q][tx + p * block_dim];
        }
    }

    // 调整循环下标
    for (size_t i = 1; i <= K / block_unit; ++i) {
        // 调整循环下标
        // 展开复杂的内层循环
        #pragma unroll
        for (size_t j = 1; j <= block_unit / thread_unit; ++j) {
            // 提前到最后一次小迭代之前，切换到下一批次共享内存
            if (j == block_unit / thread_unit) {
                if (i != K / block_unit) {
                    // 避免在共享内存使用之前被修改
                    __syncthreads();
                }
                // 切换目标缓冲区
                smem_stage_idx ^= 1;
            }
            if (!(i == K / block_unit && j == block_unit / thread_unit)) {
                // 最后一次小迭代取下一批次共享内存里的开头部分
                size_t nj = j != block_unit / thread_unit ? j : 0;
                for (size_t p = 0; p < thread_shape; ++p) {
                    for (size_t q = 0; q < thread_unit; ++q) {
                        r_a[reg_stage_idx ^ 1][p][q] =
                            s_a[smem_stage_idx][ty + p * block_dim][q + nj * thread_unit];
                    }
                }
                for (size_t q = 0; q < thread_unit; ++q) {
                    for (size_t p = 0; p < thread_shape; ++p) {
                        r_b[reg_stage_idx ^ 1][q][p] =
                            s_b[smem_stage_idx][q + nj * thread_unit][tx + p * block_dim];
                    }
                }
            }
            // 推迟到第一次小迭代的预取之后
            if (j == 1) {
                if (i != K / block_unit) {
                    // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
                    size_t col_a = i * block_unit + tx, row_b = i * block_unit + ty;
                    // 覆盖上一轮迭代计算使用的共享内存
                    for (size_t p = 0; p < thread_shape; ++p) {
                        for (size_t q = 0; tx + q * block_dim < block_unit; ++q) {
                            s_a[smem_stage_idx ^ 1][ty + p * block_dim][tx + q * block_dim] =
                                A[iy + p * block_dim][col_a + q * block_dim];
                        }
                    }
                    for (size_t q = 0; ty + q * block_dim < block_unit; ++q) {
                        for (size_t p = 0; p < thread_shape; ++p) {
                            s_b[smem_stage_idx ^ 1][ty + q * block_dim][tx + p * block_dim]
                                = B[row_b + q * block_dim][ix + p * block_dim];
                        }
                    }
                }
            }
            for (size_t k = 0; k < thread_unit; ++k) {
                for (size_t p = 0; p < thread_shape; ++p) {
                    for (size_t q = 0; q < thread_shape; ++q) {
                        sum[p][q] += r_a[reg_stage_idx][p][k] * r_b[reg_stage_idx][k][q];
                    }
                }
            }
            // 切换目标缓冲区
            reg_stage_idx ^= 1;
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