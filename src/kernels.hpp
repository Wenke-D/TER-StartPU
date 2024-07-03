#pragma once

#include <random>

#include "blas.hpp"
#include "tile.hpp"
#include "vector.hpp"
#include "log.hpp"

/**
 * This file or its .cpp file should contain all kernel implementations that are called from StarPU
 * Note : Often kernels will need to call BLAS/LAPACK,
 * an interface to these can be found in blas.hpp and blas_extensions.hpp
 *
 * Any other functions will have to be implemented
 */

// Kernels in StarPU require a very specific interface

template <typename E>
void cpu_tile_fill(void *descr[], void *arg)
{
    using Tile = Tile<E>;
    Tile t = Tile::from(descr[0]);

    std::random_device rd;                            // Obtain a random number from hardware
    std::mt19937 gen(rd());                           // Seed the generator
    std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range [0.0, 1.0]

    // Generate and print a random float
    E value = static_cast<E>(distr(gen));

    for (int j = 0; j < t.width; j++)
    {
        for (int i = 0; i < t.height; i++)
        {
            t.head[i + j * t.ld] = 1;
        }
    }
}

template <typename E>
void cpu_tile_mul(void *descr[], void *arg)
{
    using Tile = Tile<E>;
    Tile A = Tile::from(descr[0]);
    Tile B = Tile::from(descr[1]);
    Tile C = Tile::from(descr[2]);

    auto coef = Coef<E>::from(arg);

    // Log::print(__func__, A);
    // Log::print(__func__, B);
    // Log::print(__func__, C);
    // Log::print(__func__, coef);

    blas<E>::gemm(
        'N', 'N',
        C.height, C.width,
        A.width, coef.alpha,
        A.head,
        A.ld, B.head,
        B.ld, coef.beta,
        C.head, C.ld);
}