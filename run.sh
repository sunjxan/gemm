# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr cpu_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cpu_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr cpu_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cpu_2.cu -o a && ./a
