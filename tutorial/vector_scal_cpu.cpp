#include <starpu.h>
#include "cast.hpp"
#include <iostream>

/* This kernel takes a buffer and scales it by a constant factor */
void vector_scal_cpu(void *buffers[], void *cl_arg)
{
  float *head;
  size_t length;
  to_vector(&head, &length, buffers[0]);
  float factor = to_float(cl_arg);

  printf("access on %f cpu, length:%d\n", head[0], length);

  for (size_t i = 0; i < length; i++)
  {
    head[i] *= factor;
  }
}
