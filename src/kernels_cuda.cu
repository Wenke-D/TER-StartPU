
#include "executor.hpp"

__global__ void hello(float alpha, float beta, float *head)
{
    head[0] = 10;
}

__host__ void hello_cpu(float alpha, float beta, float *head)
{
    hello<<<1, 1>>>(alpha, beta, head);
}
