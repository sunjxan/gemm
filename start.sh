# echo "cpu:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o cpu_1.out && ./cpu_1.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o cpu_1_dp.out && ./cpu_1_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_2.cu -o cpu_2.out && ./cpu_2.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_2.cu -o cpu_2_dp.out && ./cpu_2_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_3.cu -o cpu_3.out && ./cpu_3.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_3.cu -o cpu_3_dp.out && ./cpu_3_dp.out
# echo ""
# echo "cuda:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_1.cu -o cuda_1.out && ./cuda_1.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_1.cu -o cuda_1_dp.out && ./cuda_1_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_2.cu -o cuda_2.out && ./cuda_2.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_2.cu -o cuda_2_dp.out && ./cuda_2_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_3.cu -o cuda_3.out && ./cuda_3.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_3.cu -o cuda_3_dp.out && ./cuda_3_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_4.cu -o cuda_4.out && ./cuda_4.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_4.cu -o cuda_4_dp.out && ./cuda_4_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_5.cu -o cuda_5.out && ./cuda_5.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_5.cu -o cuda_5_dp.out && ./cuda_5_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_6.cu -o cuda_6.out && ./cuda_6.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_6.cu -o cuda_6_dp.out && ./cuda_6_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_7.cu -o cuda_7.out && ./cuda_7.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_7.cu -o cuda_7_dp.out && ./cuda_7_dp.out
# echo ""
echo "optimize:"
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_1.cu -o optimize_1.out && ./optimize_1.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_1.cu -o optimize_1_dp.out && ./optimize_1_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_2.cu -o optimize_2.out && ./optimize_2.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_2.cu -o optimize_2_dp.out && ./optimize_2_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_3.cu -o optimize_3.out && ./optimize_3.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_3.cu -o optimize_3_dp.out && ./optimize_3_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_4.cu -o optimize_4.out && ./optimize_4.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_4.cu -o optimize_4_dp.out && ./optimize_4_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_5.cu -o optimize_5.out && ./optimize_5.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_5.cu -o optimize_5_dp.out && ./optimize_5_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_6.cu -o optimize_6.out && ./optimize_6.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_6.cu -o optimize_6_dp.out && ./optimize_6_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr optimize_7.cu -o optimize_7.out && ./optimize_7.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP optimize_7.cu -o optimize_7_dp.out && ./optimize_7_dp.out
echo ""
echo "lib:"
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cublas.cu -lcublas -o cublas.out && ./cublas.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cublas.cu -lcublas -o cublas_dp.out && ./cublas_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -I ../cutlass/include cutlass.cu -o cutlass.out && ./cutlass.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP -I ../cutlass/include cutlass.cu -o cutlass_dp.out && ./cutlass_dp.out