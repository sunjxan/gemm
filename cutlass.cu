#include <cstdio>
#include <cmath>
#include "error.cuh"
#include "cutlass/gemm/device/gemm.h"

#define DIVUP(m, n) ((m + n - 1) / n)

const unsigned SKIP = 5, REPEATS = 5;
const size_t M = 5120, N = 4096, K = 4096;
const size_t real_size = sizeof(real);
const size_t MK = M * N, KN = K * N, MN = M * N;
const size_t MK_size = MK * real_size, KN_size = KN * real_size, MN_size = MN * real_size;

void random_init(real *data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = real(rand()) / RAND_MAX;
    }
}

__global__ void check_kernel(const real *A, const real *B, real *C)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < M && iy < N) {
        real sum = 0;
        for (size_t t = 0; t < K; ++t) {
            sum += A[ix * K + t] * B[t * N + iy];
        }
        C[ix * N + iy] = sum;
    }
}

bool check(const real *A, const real *B, const real *C) {
    real *h_C;
    cudaMallocHost(&h_C, MN_size);

    real *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, MK_size);
    cudaMalloc(&d_B, KN_size);
    cudaMalloc(&d_C, MN_size);

    cudaMemcpy(d_A, A, MK_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, KN_size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, MN_size);

    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(M, block_size.x), DIVUP(N, block_size.y));
    check_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_C, d_C, MN_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            real sum = h_C[i * N + j];
            real v = C[i * N + j];
            if (std::fabs(sum - v) > EPSILON) {
                printf("C[%u][%u] not match, %.15f vs %.15f\n", unsigned(i), unsigned(j), sum, v);
                cudaFreeHost(h_C);
                return false;
            }
        }
    }
    cudaFreeHost(h_C);
    return true;
}

void gemm(const real *d_A, const real *d_B, real *d_C)
{
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<real, RowMajor, real, RowMajor, real, RowMajor>;

    CutlassGemm gemm_operator;
    
    real alpha = 1.0f, beta = 0.0f;
    CutlassGemm::Arguments args({M, N, K},
                            {d_A, K},
                            {d_B, N},
                            {d_C, N},
                            {d_C, N},
                            {alpha, beta});
    
    CHECK_CUTLASS(gemm_operator(args));
}

real timing(const real *d_A, const real *d_B, real *d_C)
{
    cudaMemset(d_C, 0, MN_size);

    float elapsed_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));

    gemm(d_A, d_B, d_C);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed_time;
}

int main()
{
    real *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, MK_size);
    cudaMallocHost(&h_B, KN_size);
    cudaMallocHost(&h_C, MN_size);

    random_init(h_A, M * K);
    random_init(h_B, K * N);

    real *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, MK_size);
    cudaMalloc(&d_B, KN_size);
    cudaMalloc(&d_C, MN_size);

    cudaMemcpy(d_A, h_A, MK_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, KN_size, cudaMemcpyHostToDevice);

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(d_A, d_B, d_C);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(d_A, d_B, d_C);
        total_time += elapsed_time;
    }
    printf("Time: %.3f ms\n", total_time / REPEATS);

    cudaMemcpy(h_C, d_C, MN_size, cudaMemcpyDeviceToHost);
    printf("Check: %s\n", check(h_A, h_B, h_C) ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 0;
}
