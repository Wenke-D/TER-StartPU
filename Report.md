# TER StarPU GEMM

## Runtime system comparision
Starpu
MAGMA
HPX
Charm++

OpenMP
C++ native support for parallelisme
CUDA


## Code design
When modeling the matrix, we have 4 classes for different aspect of the matrix depends on the usage. `MatrixDescriptor`, `Matrix`, `Tiles`, `Tile`.

`MatrixDescriptor`: Defines how a matrix is stored in the memory. It describes the matrix's memeory layout, width, height, leading dimension and header pointer. This classes assumes column major storage of the data.

`Matrix`: This class is in charge of initilzation and memory management. It can allocate memories, assign values to the memory and deallocate the memory. This class is to ensure a validate matrix creation. It also contains shortcut for tiling a matrix.

`Tiles`: This class is for tiling a matrix into tiles, this class contain information on how the metric of the tiling, such as dimension of a tile, how many tiles the matrix contains in a row/column. It also contains API for opeartion on all the tiles.

`Tile`: is the basic computation unit of a matrix, any matrix computation will eventually transforme into computation of tiles. This class contains information of the location and character of its data. The behavior of each tile is also defined here. As in this project, execution is based on StarPU, and every data unit can only belongs to one StarPU data handler, this class holds a data_handler to simplify programming.


## performance
### histry based model
4096 x 4096 matrix, 1024 tile size give best result
for "bad" sizes, dmda is similar in performance to lws but for "good" sizes dmda becomes much better than lws
### 