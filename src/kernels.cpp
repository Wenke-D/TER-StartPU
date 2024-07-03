#include "kernels.hpp"

// instantiate template for float and double
template void cpu_tile_mul<float>(void *descr[], void *arg);
template void cpu_tile_mul<double>(void *descr[], void *arg);
