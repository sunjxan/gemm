#include <cstdio>

#include "common.hpp"

// 双缓冲ping-pong，原理是指令级并行，如果指令之间相互独立，没有读写依赖，
// 线程读写操作的顺序和指令在代码中出现的顺序不一定相同
// __syncthreads函数可以确保该线程块在障碍点之前的读写操作已经完成
// __threadfence_block函数不进行障碍同步，只是对当前线程之前的写入操作进行刷新
// __threadfence_block/__threadfence/__threadfence_system对应刷新的内存和缓存层次不断提升

// block_shape应能整除M、K、N，unit应能整除K
// unit取简单情况等于block_shape，一次取满
constexpr size_t block_shape = 32, unit = block_shape;

__global__ void kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{
    unsigned ty = threadIdx.y, iy = blockIdx.y * block_shape + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * block_shape + tx;

    __shared__ real s_a[2][block_shape][unit], s_b[2][unit][block_shape];

    real sum = 0.0f;

    unsigned smem_stage_idx = 0;
    // 安培之前的架构，从全局内存转移到共享内存会经过寄存器中转
    s_a[smem_stage_idx][ty][tx] = A[iy][tx];
    s_b[smem_stage_idx][ty][tx] = B[ty][ix];
    __syncthreads();

    for (size_t i = 1; i <= K / unit; ++i) {
        // 避免在共享内存使用之前被修改
        if (i != 1) {
            __syncthreads();
            smem_stage_idx ^= 1;
        }
        if (i != K / unit) {
            // 在A中拷贝的列序col_a，在B中拷贝的行序row_b
            size_t i_unit = i * unit, col_a = i_unit + tx, row_b = i_unit + ty;
            // 覆盖上一个迭代计算使用的共享内存
            s_a[smem_stage_idx ^ 1][ty][tx] = A[iy][col_a];
            s_b[smem_stage_idx ^ 1][ty][tx] = B[row_b][ix];
        }
        for (size_t j = 0; j < unit; ++j) {
            sum += s_a[smem_stage_idx][ty][j] * s_b[smem_stage_idx][j][tx];
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
