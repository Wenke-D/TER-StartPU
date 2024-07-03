
#include <starpu.h>
#include <stdlib.h>
#include <iostream>
#include "cast.hpp"

void to_vector(float **head, size_t *length, void *buffer)
{
    *length = STARPU_VECTOR_GET_NX(buffer);
    *head = (float *)STARPU_VECTOR_GET_PTR(buffer);
}

float to_float(void *cl_arg)
{
    float *ptr = (float *)cl_arg;
    return *ptr;
}