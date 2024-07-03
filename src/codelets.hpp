#pragma once
#include <string>
#include <sstream>
#include "executor.hpp"

// forward declartion
template <typename E>
void cpu_tile_mul(void *descr[], void *arg);

template <typename E>
void cpu_array_fill(void *descr[], void *arg);

template <typename E>
void cpu_tile_fill(void *descr[], void *arg);

template <typename E>
void cuda_gemm(void *buffers[], void *_args);

/**
 * This file or its .cpp file should contain the implementation of all codelets needed.
 */

/**
 * A global static codelet based on the data type.
 *
 * Note: when using a local codelet in a function, beware of the the life scope.
 * When register a local codelet to a task, then manage the task at outside of the function,
 * the codelet is destoried, so task has a dangling pointer.
 */

template <typename DataType>
struct Coef
{
    DataType alpha;
    DataType beta;

    static Coef<DataType> from(void *cl_arg)
    {
        using E = Coef<DataType>;
        E *ptr = (E *)cl_arg;
        return *ptr;
    }

    std::string toString()
    {
        std::ostringstream ss;
        ss << "alpha: " << alpha << ", beta: " << beta;
        return ss.str();
    }
};

static struct starpu_perfmodel tile_mul_perf_model_history =
    {
        .type = STARPU_HISTORY_BASED,
        .symbol = "tile_mul_perf_model_history"};

static struct starpu_perfmodel tile_mul_perf_model_regression =
    {
        .type = STARPU_HISTORY_BASED,
        .symbol = "tile_mul_perf_model_regression"};

template <typename DataType>
struct tile_mul
{
    static starpu_codelet &codelet()
    {
        static starpu_codelet cl = {
            .cpu_funcs = {cpu_tile_mul<DataType>},
            .cuda_funcs = {cuda_gemm<DataType>},
            .nbuffers = 3,
            .modes = {STARPU_R, STARPU_R, STARPU_RW},
            .model = &tile_mul_perf_model_regression};
        return cl;
    }
};

template <typename DataType>
struct tile_fill
{
    static starpu_codelet &codelet()
    {
        static starpu_codelet cl = {
            .cpu_funcs = {cpu_tile_fill<DataType>},
            .nbuffers = 1,
            .modes = {STARPU_W},
        };
        return cl;
    }
};