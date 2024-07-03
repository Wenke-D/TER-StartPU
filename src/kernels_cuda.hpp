
#pragma once
#include "executor.hpp"
#include "tile.hpp"
#include "log.hpp"

void hello_cpu(float alpha, float beta, float *head);

template <typename E>
void cuda_gemm(void *buffers[], void *_args)
{
    Tile<E> A = Tile<E>::from(buffers[0]);
    Tile<E> B = Tile<E>::from(buffers[1]);
    Tile<E> C = Tile<E>::from(buffers[2]);

    Coef<E> coef = Coef<E>::from(_args);

    int m = C.height;
    int n = C.width;
    int k = A.width;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    cublasStatus_t status = cublasSgemm(
        starpu_cublas_get_local_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &coef.alpha, A.head, A.ld, B.head, B.ld,
        &coef.beta, C.head, C.ld);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        STARPU_CUDA_REPORT_ERROR(err);
}
