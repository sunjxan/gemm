#include <cstdio>

#include "common.cuh"
#include "cutlass/gemm/device/gemm.h"

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

int main()
{
    launch_gpu();
    return 0;
}
