#pragma once

#include "executor.hpp"

/**
 * Memory layout of a matrix.
 */
template <typename T>
class MatrixDescriptor
{
public:
    T *head;
    const int width;
    const int height;
    const int ld;

private:
    MatrixDescriptor(T *head,
                     const int width,
                     const int height,
                     const int ld);

public:
    /**
     * Setup a descriptpr including allocate memory space.
     */
    static MatrixDescriptor create(int width, int height);

    /**
     * Deallocate the space
     */
    void free();
};

template <typename T>
MatrixDescriptor<T>::MatrixDescriptor(
    T *head, const int width, const int height, const int ld)
    : head(head), width(width), height(height), ld(ld){};

template <typename T>
MatrixDescriptor<T> MatrixDescriptor<T>::create(int width, int height)
{
    size_t data_size = width * height * sizeof(T);
    T *head;
    starpu_malloc((void **)&head, data_size);
    return MatrixDescriptor{head,
                            width,
                            height,
                            height};
}

template <typename T>
void MatrixDescriptor<T>::free()
{
    starpu_free(head);
}
