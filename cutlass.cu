#include "common.hpp"
#include "cutlass/gemm/device/gemm.h"

void matmul(const real *A, const real *B, real *C)
{
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<real, RowMajor, real, RowMajor, real, RowMajor>;

    CutlassGemm gemm_operator;
    
    real alpha = 1.0, beta = 0.0;
    CutlassGemm::Arguments args({M, N, K},
                            {A, K},
                            {B, N},
                            {C, N},
                            {C, N},
                            {alpha, beta});
    
    CHECK_CUTLASS(gemm_operator(args));
}

int main()
{
    launch_gpu();
    return 0;
}