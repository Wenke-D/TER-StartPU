#pragma once
#include <iostream>
#include <sstream>
#include "codelets.hpp"
#include "executor.hpp"

// forward signature to resolve cyclic dependancy between "tile.hpp" and "kernels.hpp"
template <typename E>
void cpu_tile_mul(void *descr[], void *arg);

/**
 * The tile class is an abstraction to hold StarPU matrix interfaces and launch tasks on them.
 * It proprety is the same as the Matrix class, except it contains a starpu_data_handler.
 */
template <typename DataType>
struct Tile
{
	DataType *head;
	int width;
	int height;
	int ld;
	starpu_data_handle_t handler;

	Tile(DataType *head, int width, int height, int ld)
		: head(head), width(width), height(height), ld(ld){};

	/**
	 * Register this tile to its own handler
	 */
	void checkin()
	{
		size_t element_size = sizeof(DataType);
		starpu_matrix_data_register(
			&handler, STARPU_MAIN_RAM,
			(uintptr_t)head, (u_int32_t)ld,
			(u_int32_t)height,
			(u_int32_t)width, element_size);
	}

	DataType at(int i, int j)
	{
		return head[j * ld + i];
	}

	/**
	 * Unregister this tile to its own handler
	 */
	void checkout()
	{
		starpu_data_unregister(handler);
	}

	std::string toString()
	{
		std::ostringstream oss;
		oss << "Width: " << width << ", Height: " << height << ", Leading Dimension: " << ld << "\n";

		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				oss << at(i, j) << " ";
			}
			oss << "\n";
		}
		return oss.str();
	}

	void print()
	{
		std::ostringstream oss;

		oss << "==========\n";
		oss << "Tile:\nWidth: " << width << ", Height: " << height << ", Leading Dimension: " << ld << "\n";

		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				oss << at(i, j) << " ";
			}
			oss << "\n";
		}

		oss << "==========\n";
		std::cout << oss.str();
	};

	starpu_data_handle_t &data_handler()
	{
		return handler;
	}

	static Tile<DataType> from(void *descr)
	{
		// The recovery is transposed because registeration is transposed.
		DataType *head = (DataType *)STARPU_MATRIX_GET_PTR(descr);
		int width = (int)STARPU_MATRIX_GET_NY(descr);
		int height = (int)STARPU_MATRIX_GET_NX(descr);
		int ld = (int)STARPU_MATRIX_GET_LD(descr);
		return Tile<DataType>{head, width, height, ld};
	}

	/**
	 * [async]
	 * Filling this tile by random values.
	 */
	void random()
	{
		starpu_task *tk = starpu_task_create();
		tk->cl = &tile_fill<DataType>::codelet();

		tk->handles[0] = this->data_handler();
		int ret = starpu_task_submit(tk);
		STARPU_CHECK_RETURN_VALUE(ret, "tile_fill_task_submit");
	}

	/**
	 * [async]
	 *
	 * All tiles are expected to already checkin.
	 */
	static void gemm(
		Tile<DataType> &A,
		Tile<DataType> &B,
		Tile<DataType> &C, Coef<DataType> coef)
	{

		starpu_task *tk = starpu_task_create();
		tk->cl = &tile_mul<DataType>::codelet();
		tk->cl_arg = static_cast<void *>(&coef);
		tk->cl_arg_size = sizeof(Coef<DataType>);

		/*
		 * Warning: we copy the handler here and put it into task,
		 * It may cause problem when unregister the handler at outside.
		 * In case of error, change it to reference type.
		 */
		starpu_data_handle_t hdA = A.handler;
		starpu_data_handle_t hdB = B.handler;
		starpu_data_handle_t hdC = C.handler;

		tk->handles[0] = hdA;
		tk->handles[1] = hdB;
		tk->handles[2] = hdC;

		int ret = starpu_task_submit(tk);
		STARPU_CHECK_RETURN_VALUE(ret, "task_submit");
	}
};
