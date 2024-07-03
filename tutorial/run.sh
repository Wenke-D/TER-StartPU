
set -e

nvcc $(pkg-config --cflags starpu-1.4) -g -c vector_scal_cpu.cpp -o vector_scal_cpu.o
nvcc $(pkg-config --cflags starpu-1.4) -g -c vector_scal_main.cpp -o vector_scal_main.o
nvcc $(pkg-config --cflags starpu-1.4) -g -c vector_scal_cuda.cu -o vector_scal_cuda.o
nvcc $(pkg-config --cflags starpu-1.4) -g -c cast.cpp -o cast.o

nvcc $(pkg-config --libs starpu-1.4) cast.o vector_scal_cpu.o vector_scal_main.o vector_scal_cuda.o -o scale

./scale