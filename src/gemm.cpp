#include "headers_app.hpp"
#include "reference_code.hpp"
#include "profiling.hpp"

void printHelp()
{
  std::cout << "Parameters for gemm:\n"
               "  --help            --  Print this help\n"
               "  --type [d/s]      --  Set the data type (double/single)\n"
               "  --m [uint]        --  Set the size of M\n"
               "  --n [uint]        --  Set the size of N\n"
               "  --k [uint]        --  Set the size of K\n"
               "  --bs [uint]       --  Set the block size\n"
               "  --transA [N/T]    --  Set transposition value for A\n"
               "  --transB [N/T]    --  Set transposition value for B\n"
               "  --alpha [float]   --  Set value for alpha\n"
               "  --beta [float]    --  Set value for beta\n";
#ifdef USE_CUDA
  std::cout << "  --gpu             --  Use available GPUs\n";
#endif
#ifdef USE_MPI
  std::cout << "  --P [uint]        --  Set the process grid row number\n"
               "  --Q [uint]        --  Set the process grid column number\n"
               "  --stat [A/B/C]    --  Set the stationary matrix\n"
               "  --redux           --  Use MPI reductions\n";
#endif
}

int main(int argc, char **argv)
{
  arg_parser parser(argc, argv);

  if (parser.get("--help"))
  {
    printHelp();
    exit(0);
  }

  // [[maybe_unused]] auto type = parser.get<char>("--type", 's'); // Data type used for computations
  [[maybe_unused]] auto m = parser.get<unsigned>("--m", 1024);         // Length of the M dimension for matrices A and C
  [[maybe_unused]] auto n = parser.get<unsigned>("--n", 1024);         // Length of the N dimension for matrices B and C
  [[maybe_unused]] auto k = parser.get<unsigned>("--k", 1024);         // Length of the K dimension for matrices A and B
  [[maybe_unused]] auto factor = parser.get<unsigned>("--factor", 10); // factor of matrix and tile size
  // [[maybe_unused]] auto bs = parser.get<unsigned>("--bs", 256); // Length of dimensions for the tiles
  // [[maybe_unused]] auto transA = parser.get<char>("--transA", 'N'); // Value of transposition for A, can be 'N' or 'T'
  // [[maybe_unused]] auto transB = parser.get<char>("--transB", 'N'); // Value of transposition for B, can be 'N' or 'T'
  // [[maybe_unused]] auto alpha = parser.get<float>("--alpha", 1.0);  // Value for the alpha parameter (note : C = alpha * A * B + beta * C)
  // [[maybe_unused]] auto beta = parser.get<float>("--beta", 0.0);    // Value for the beta parameter
#ifdef USE_CUDA
  [[maybe_unused]] auto enable_gpu = parser.get("--gpu"); // If activated, use available GPUs
#endif
#ifdef USE_MPI
  [[maybe_unused]] auto process_rows = parser.get<unsigned>("--P", 1); // MPI process grid row size
  [[maybe_unused]] auto process_cols = parser.get<unsigned>("--Q", 1); // MPI process grid column size
  [[maybe_unused]] auto stat = parser.get<char>("--stat", 'C');        // Stationary matrix
  [[maybe_unused]] auto redux = parser.get("--redux");                 // If activated, use MPI reductions
#endif

  int ret = starpu_init(nullptr);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
  // Initialize cuBLAS
  starpu_cublas_init();

  int M = m;
  int N = n;
  int K = k;

  MatrixDescriptor<float> dA = MatrixDescriptor<float>::create(K, M);
  MatrixDescriptor<float> dB = MatrixDescriptor<float>::create(N, K);
  MatrixDescriptor<float> dC = MatrixDescriptor<float>::create(N, M);

  MatrixDescriptor<float> dC_ref = MatrixDescriptor<float>::create(N, M);

  int tile_width = n / factor;
  int tile_height = n / factor;

  printf("tile size %d\n", tile_width);

  Matrix<float> A = Matrix<float>::of(dA, tile_width, tile_height);
  Matrix<float> B = Matrix<float>::of(dB, tile_width, tile_height);
  Matrix<float> C = Matrix<float>::of(dC, tile_width, tile_height);

  Matrix<float> C_ref = Matrix<float>::of(dC_ref, tile_width, tile_height);

  A.random();
  B.random();

  starpu_task_wait_for_all();

  Coef<float> coef{1, 1};

  auto seqGemm = [&]()
  { gemm_seq(A, B, C_ref, coef); };
  measureExecutionTime("seq gemm", seqGemm);

  auto asyncGemm = [&]()
  {
    Matrix<float>::gemm(A, B, C, coef);
    starpu_task_wait_for_all();
  };
  measureExecutionTime("async gemm", asyncGemm);

  C.unregister();

  std::cout << (C.equals(C_ref) ? "correct!" : "wrong!") << std::endl;

  starpu_cublas_shutdown();
  starpu_shutdown();

  return 0;
}
