/**
 * This file contains header files of executors, StarPU and Cuda.
 */

#include <starpu.h>
#ifdef USE_MPI
#include <mpi.h>
#include <starpu_mpi.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <starpu_cublas_v2.h>
#endif
