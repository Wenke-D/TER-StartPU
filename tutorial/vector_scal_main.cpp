/**
 * Tutorial: Complete the parts marked TODO in the source files
 * 1) Scaling a vector using 1 task
 * 2) Scaling a vector using a tiling and multiple tasks
 *
 * To compile the program:
 * .c files -> gcc $(pkg-config --cflags starpu-1.4) -c file.c
 * .cu files -> nvcc $(pkg-config --cflags starpu-1.4) -c file.cu
 * .o files -> nvcc $(pkg-config --libs starpu-1.4) file1.o file2.o [...] -o scale
 *
 * To go further:
 * - Look at htop when the program runs. What happens? Why?
 * - Test some environment variables of StarPU e.g.
 *   STARPU_WORKER_STATS STARPU_NCPU STARPU_NCUDA
 *   (execute with ENVIRONMENT_VARIABLE=VALUE ./scale)
 * - Test replacing malloc with starpu_malloc
 */

#include <starpu.h>
#include <iostream>

#define NX 10
#define PAR 2

void vector_scal_cpu(void *buffers[], void *_args);
void vector_scal_cuda(void *buffers[], void *_args);

static struct starpu_perfmodel perfmodel = {
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "vector_scal"};

int main(void)
{
	double start_time;

	struct starpu_codelet vec_scal_cl =
		{
			.where = STARPU_CUDA | STARPU_CPU,
			.cpu_funcs = {vector_scal_cpu},
			.cuda_funcs = {vector_scal_cuda},
			.nbuffers = 1,
			.modes = {STARPU_RW},
			.model = &perfmodel,
		};

	// Initialize StarPU with default configuration
	starpu_init(NULL);

	float *vector = new float[NX];
	for (unsigned i = 0; i < NX; i++)
		vector[i] = 1;

	constexpr long SIZE = NX;

	// fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);

	// starpu_data_handle_t vector_handle;
	// starpu_vector_data_register(
	// 	&vector_handle,
	// 	STARPU_MAIN_RAM,
	// 	(uintptr_t)vector,
	// 	SIZE,
	// 	sizeof(float));

	float factor = 3.14;

	// start_time = starpu_timing_now();

	// starpu_task *task_vec_scal = starpu_task_create();
	// task_vec_scal->cl = &vec_scal_cl;
	// task_vec_scal->cl_arg = static_cast<void *>(&factor);
	// task_vec_scal->cl_arg_size = sizeof(factor);
	// task_vec_scal->handles[0] = vector_handle;

	// starpu_task_submit(task_vec_scal);

	// starpu_task_wait_for_all();

	// fprintf(stderr, "computation took %fµs\n", starpu_timing_now() - start_time);

	// starpu_data_unregister(vector_handle);

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);

	starpu_data_handle_t vector_handles[PAR];
	starpu_task *tasks[PAR];

	size_t sub_length = SIZE / PAR;

	start_time = starpu_timing_now();

	for (size_t i = 0; i < PAR; i++)
	{
		// register data
		starpu_data_handle_t &hd = vector_handles[i];
		int offset = i * sub_length;
		starpu_vector_data_register(
			&hd,
			STARPU_MAIN_RAM,
			(uintptr_t)(vector + offset),
			sub_length,
			sizeof(float));

		// setup task
		starpu_task *tk_ptr = tasks[i];
		tk_ptr = starpu_task_create();
		tk_ptr->cl = &vec_scal_cl;
		tk_ptr->cl_arg = static_cast<void *>(&factor);
		tk_ptr->cl_arg_size = sizeof(factor);
		tk_ptr->handles[0] = hd;

		// submit task
		starpu_task_submit(tk_ptr);
	}

	starpu_task_wait_for_all();

	fprintf(stderr, "computation took %fµs\n",
			starpu_timing_now() - start_time);

	for (size_t i = 0; i < PAR; i++)
	{
		// unregister data
		starpu_data_unregister(vector_handles[i]);
	}

	fprintf(stderr, "AFTER PARALLEL First element is %f\n", vector[0]);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
	return 0;
}
