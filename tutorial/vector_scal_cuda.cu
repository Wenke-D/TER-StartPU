#include "cast.hpp"
#include <starpu.h>

static __global__ void vector_mult_cuda(float *head, long length, float factor)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length)
    head[i] *= factor;
}

void vector_scal_cuda(void *buffers[], void *_args)
{
  float factor = to_float(_args);

  float *head;
  size_t length;
  to_vector(&head, &length, buffers[0]);

  printf("access on %f gpu\n", head[0]);

  unsigned threads_per_block = 64;
  unsigned nblocks = (length + threads_per_block - 1) / threads_per_block;

  vector_mult_cuda<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(
      head,
      length,
      factor);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
