
#pragma once
#include <vector>

#include "tile.hpp"
#include "MatrixDescriptor.hpp"

template <typename T>
bool real_equals(T a, T b, T epsilon = 1e-6)
{
    return std::fabs(a - b) < epsilon;
}

/**
 * Logical representation of all tiles in a matrix.
 * As Tile has state on starPU property. This class is also stateful. One should
 * not create more than one Tiles instance on a Matrix instance.
 */
template <typename E>
class Matrix
{
private:
    std::vector<Tile<E>> tiles;

    /**
     * Width of each tile.
     */
    const int width;
    /**
     * Height of each tile.
     */
    const int height;

    Matrix(
        std::vector<Tile<E>> tiles,
        int width, int height,
        int row_count, int column_count)
        : tiles(tiles),
          width(width), height(height),
          row_count(row_count), column_count(column_count){};

public:
    /**
     * Number of tiles in vertical direction.
     */
    const int row_count;
    /**
     * Number of tiles in horizontal direction.
     */
    const int column_count;

    int matrix_width()
    {
        return column_count * width;
    };
    int matrix_height()
    {
        return row_count * height;
    }
    E *head()
    {
        return tiles[0].head;
    }

    /**
     * Create a tiled matrix.
     * @param dim dimension of the matrix
     * @param width width of each tile
     * @param height height of each tile
     */
    static Matrix<E> of(MatrixDescriptor<E> dim, int width, int height)
    {
        if (dim.width % width != 0 || dim.height % height != 0)
        {
            throw std::invalid_argument("Matrix dim is not a multiple of tile dim");
        }
        int row_count = dim.height / height;
        int column_count = dim.width / width;

        std::vector<Tile<E>> tiles;

        for (int j = 0; j < column_count; j++)
        {
            for (int i = 0; i < row_count; i++)
            {
                int x = j * width;
                int y = i * height;
                int offset = y + x * dim.ld;
                E *head = dim.head + offset;
                tiles.push_back(Tile<E>{head, width, height, dim.height});
            }
        }

        for (auto &e : tiles)
        {
            e.checkin();
        }

        return Matrix<E>{tiles, width, height, row_count, column_count};
    }

    /**
     * Get a tile by its index in the matrix.
     * @param i row index, y direction
     * @param j column index, x direction
     */
    Tile<E> &tile_at(int i, int j)
    {
        int index = j * row_count + i;
        return tiles[index];
    }

    E at(int i, int j)
    {
        Tile<E> &t = tile_at(i / height, j / width);
        return t.at(i % height, j % width);
    }

    template <typename F>
    void foreach (F f)
    {
        for (Tile<E> &e : tiles)
        {
            f(e);
        }
    }

    /**
     * [async]
     */
    void random()
    {
        for (Tile<E> &e : tiles)
        {
            e.random();
        }
    }

    std::string toString()
    {
        std::ostringstream oss;

        for (int i = 0; i < matrix_height(); ++i)
        {
            for (int j = 0; j < matrix_width(); ++j)
            {
                oss << at(i, j) << " ";
            }
            oss << "\n";
        }
        return oss.str();
    }

    /**
     * Asynchronous function.
     * Compute gemm of 3 groups of tiles using starPU for gemm.
     */
    static void gemm(
        Matrix<E> &A,
        Matrix<E> &B,
        Matrix<E> &C, Coef<E> coef)
    {

        Coef<E> first_coef{coef.alpha, coef.beta};
        Coef<E> other_coef{coef.alpha, 1};

        int tile_M = C.row_count;
        int tile_N = C.column_count;
        int tile_K = A.column_count;

        for (int i = 0; i < tile_M; i++)
        {
            for (int j = 0; j < tile_N; j++)
            {
                for (int k = 0; k < tile_K; k++)
                {
                    Tile<E> &tA = A.tile_at(i, k);
                    Tile<E> &tB = B.tile_at(k, j);
                    Tile<E> &tC = C.tile_at(i, j);
                    auto coef = (k == 0 ? first_coef : other_coef);
                    Tile<E>::gemm(tA, tB, tC, coef);
                }
            }
        }
    }

    bool equals(Matrix<E> &other)
    {
        if (this->matrix_height() != other.matrix_height() || this->matrix_width() != other.matrix_width())
        {
            throw std::invalid_argument("Comparing matrix with different dimension");
        }

        for (int i = 0; i < matrix_height(); ++i)
        {
            for (int j = 0; j < matrix_width(); ++j)
            {
                float x = at(i, j);
                float y = other.at(i, j);
                if (!real_equals(x, y))
                {
                    printf("wrong at (%d, %d), %f != %f\n", i, j, x, y);
                    return false;
                }
            }
        }
        return true;
    }

    void unregister()
    {
        for (auto &e : tiles)
        {
            e.checkout();
        }
    }
};