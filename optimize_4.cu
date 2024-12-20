#include "common.hpp"

// 将矩阵A的共享内存和寄存器片段转置存储，便于从共享内存传输到寄存器也使用指令LDS.128优化

#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

// block_shape应能整除M、K、N，block_unit应能整除K，block_unit应是4的正整数倍
constexpr size_t block_shape = 128, block_unit = 8;
// thread_shape应能整除block_shape和block_unit，thread_shape应是4的正整数倍
constexpr size_t thread_shape = 8, block_dim = block_shape / thread_shape;
// thread_unit应能整除block_unit
constexpr size_t thread_unit = 1;

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{
    unsigned tid = threadIdx.x, ty = tid / block_dim, tx = tid % block_dim;
    unsigned by = blockIdx.y * block_shape, bx = blockIdx.x * block_shape;

    size_t bit = real_size == sizeof(float) ? 2 : 1;
    constexpr size_t trans_size = real_size == sizeof(float) ? 4 : 2;

    __shared__ real s_a[2][block_unit][block_shape], s_b[2][block_unit][block_shape];

    // 不是协同拷贝，每个线程拷贝自己所需的数据，不同线程会重复拷贝数据
    real r_a[2][thread_unit][thread_shape], r_b[2][thread_unit][thread_shape];

    real sum[thread_shape][thread_shape];
    for (size_t p = 0; p < thread_shape; ++p) {
        for (size_t q = 0; q < thread_shape; q+=trans_size) {
            FLOAT4(sum[p][q]) = float4{0.0};
        }
    }

    // 取第一部分
    unsigned smem_stage_idx = 0;
    real trans[trans_size];
    for (size_t i = tid << bit; i < block_shape * block_unit; i += blockDim.x << bit) {
        size_t j = i / block_unit, k = i % block_unit;
        // trans做矩阵A转置的中转寄存器数组
        FLOAT4(trans[0]) = CFLOAT4(A[j + by][k]);
        for (size_t x = 0; x < trans_size; ++x) {
            s_a[smem_stage_idx][k + x][j] = trans[x];
        }

        k = i / block_shape, j = i % block_shape;
        FLOAT4(s_b[smem_stage_idx][k][j]) = CFLOAT4(B[k][j + bx]);
    }
    // 协同拷贝，等待拷贝结束
    __syncthreads();

    // 取第一部分
    unsigned reg_stage_idx = 0;
    for (size_t q = 0; q < thread_unit; ++q) {
        for (size_t p = 0; p < thread_shape; p+=trans_size) {
            FLOAT4(r_a[reg_stage_idx][q][p]) = FLOAT4(s_a[smem_stage_idx][q][ty * trans_size + p * block_dim]);
            FLOAT4(r_b[reg_stage_idx][q][p]) = FLOAT4(s_b[smem_stage_idx][q][tx * trans_size + p * block_dim]);
        }
    }

    // 调整循环下标
    for (size_t i = 1; i <= K / block_unit; ++i) {
        // 调整循环下标
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
                for (size_t q = 0; q < thread_unit; ++q) {
                    for (size_t p = 0; p < thread_shape; p+=trans_size) {
                        FLOAT4(r_a[reg_stage_idx ^ 1][q][p]) =
                            FLOAT4(s_a[smem_stage_idx][q + nj * thread_unit][ty * trans_size + p * block_dim]);
                        FLOAT4(r_b[reg_stage_idx ^ 1][q][p]) =
                            FLOAT4(s_b[smem_stage_idx][q + nj * thread_unit][tx * trans_size + p * block_dim]);
                    }
                }
            }
            // 推迟到第一次小迭代的预取之后
            if (j == 1) {
                if (i != K / block_unit) {
                    // 覆盖上一轮迭代计算使用的共享内存
                    for (size_t r = tid << bit; r < block_shape * block_unit; r += blockDim.x << bit) {
                        size_t s = r / block_unit, t = r % block_unit;
                        // trans做矩阵A转置的中转寄存器数组
                        FLOAT4(trans[0]) = CFLOAT4(A[s + by][t + i * block_unit]);
                        for (size_t x = 0; x < trans_size; ++x) {
                            s_a[smem_stage_idx ^ 1][t + x][s] = trans[x];
                        }

                        t = r / block_shape, s = r % block_shape;
                        FLOAT4(s_b[smem_stage_idx ^ 1][t][s]) = CFLOAT4(B[t + i * block_unit][s + bx]);
                    }
                }
            }
            for (size_t k = 0; k < thread_unit; ++k) {
                for (size_t p = 0; p < thread_shape; ++p) {
                    for (size_t q = 0; q < thread_shape; ++q) {
                        sum[p][q] += r_a[reg_stage_idx][k][p] * r_b[reg_stage_idx][k][q];
                    }
                }
            }
            // 切换目标缓冲区
            reg_stage_idx ^= 1;
        }
    }
    // 写入位置也变了，因为操作的数据位置变了
    size_t row_start = by + ty * trans_size, col_start = bx + tx * trans_size;
    for (size_t p = 0; p < thread_shape; p+=trans_size) {
        for (size_t y = 0; y < trans_size; ++y) {
            for (size_t q = 0; q < thread_shape; q+=trans_size) {
                FLOAT4(C[row_start + p * block_dim + y][col_start + q * block_dim]) = FLOAT4(sum[p + y][q]);
            }
        }
    }
}

void matmul(const real *A, const real *B, real *C)
{
    const real (*nA)[K] = reinterpret_cast<decltype(nA)>(A);
    const real (*nB)[N] = reinterpret_cast<decltype(nB)>(B);
    real (*nC)[N] = reinterpret_cast<decltype(nC)>(C);

    // 线程块是正方形
    dim3 block_size(block_dim * block_dim);
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