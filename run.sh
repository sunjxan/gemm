# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr cpu_1.cu -o a && ./a
Time: 2619.040 ms
Check: OK
# nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cpu_1.cu -o a && ./a
Time: 2534.317 ms
Check: OK
