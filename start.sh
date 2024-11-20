nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr cuda_1.cu -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_1.cu -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr cublas.cu -lcublas -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cublas.cu -lcublas -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -I cutlass/include cutlass.cu -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP -I cutlass/include cutlass.cu -o a && ./a
