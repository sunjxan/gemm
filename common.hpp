#pragma once
#include <cstdio>
#include <cmath>

#include "error.h"

constexpr unsigned SKIP = 5, REPEATS = 5;
constexpr size_t M = 5120, N = 2048, K = 4096;
constexpr size_t real_size = sizeof(real);
constexpr size_t MK = M * K, KN = K * N, MN = M * N;
constexpr size_t MK_size = MK * real_size, KN_size = KN * real_size, MN_size = MN * real_size;

void matmul(const real *, const real *, real *);

void random_init(real *data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = real(rand()) / RAND_MAX;
    }
}

__global__ void check_kernel(const real (*A)[K], const real (*B)[N], real (*C)[N])
{
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < M && ix < N) {
        real sum = 0.0;
        for (size_t t = 0; t < K; ++t) {
            sum += A[iy][t] * B[t][ix];
        }
        C[iy][ix] = sum;
    }
}

bool check(const real *A, const real *B, const real *C) {
    real *h_C = nullptr;
    CHECK(cudaMallocHost(&h_C, MN_size));

    real *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(&d_A, MK_size));
    CHECK(cudaMalloc(&d_B, KN_size));
    CHECK(cudaMalloc(&d_C, MN_size));

    CHECK(cudaMemcpy(d_A, A, MK_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, KN_size, cudaMemcpyHostToDevice));

    const real (*d_nA)[K] = reinterpret_cast<decltype(d_nA)>(d_A);
    const real (*d_nB)[N] = reinterpret_cast<decltype(d_nB)>(d_B);
    real (*d_nC)[N] = reinterpret_cast<decltype(d_nC)>(d_C);

    dim3 block_size(32, 32);
    // N是列对应x，M是行对应y
    dim3 grid_size(DIVUP(N, block_size.x), DIVUP(M, block_size.y));
    check_kernel<<<grid_size, block_size>>>(d_nA, d_nB, d_nC);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_C, d_C, MN_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t pos = i * N + j;
            real sum = h_C[pos], v = C[pos];
            if (std::fabs(sum - v) > EPSILON) {
                printf("C[%u][%u] not match, %.15f vs %.15f\n", unsigned(i), unsigned(j), sum, v);
                CHECK(cudaFreeHost(h_C));
                return false;
            }
        }
    }
    CHECK(cudaFreeHost(h_C));
    return true;
}

real timing(const real *A, const real *B, real *C)
{
    float elapsed_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));

    matmul(A, B, C);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed_time;
}

void launch_cpu()
{
    real *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    CHECK(cudaMallocHost(&h_A, MK_size));
    CHECK(cudaMallocHost(&h_B, KN_size));
    CHECK(cudaMallocHost(&h_C, MN_size));

    random_init(h_A, MK);
    random_init(h_B, KN);

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(h_A, h_B, h_C);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(h_A, h_B, h_C);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    printf("Check: %s\n", check(h_A, h_B, h_C) ? "OK" : "Failed");

    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));
}

void launch_gpu()
{
    real *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    CHECK(cudaMallocHost(&h_A, MK_size));
    CHECK(cudaMallocHost(&h_B, KN_size));
    CHECK(cudaMallocHost(&h_C, MN_size));

    random_init(h_A, MK);
    random_init(h_B, KN);

    real *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(&d_A, MK_size));
    CHECK(cudaMalloc(&d_B, KN_size));
    CHECK(cudaMalloc(&d_C, MN_size));

    CHECK(cudaMemcpy(d_A, h_A, MK_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, KN_size, cudaMemcpyHostToDevice));

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(d_A, d_B, d_C);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(d_A, d_B, d_C);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    CHECK(cudaMemcpy(h_C, d_C, MN_size, cudaMemcpyDeviceToHost));
    printf("Check: %s\n", check(h_A, h_B, h_C) ? "OK" : "Failed");

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));
}
