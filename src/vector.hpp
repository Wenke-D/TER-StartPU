#pragma once

#include <starpu.h>
#include <iostream>
#include <sstream>

template <typename E>
struct Vector
{
    E *head;
    int length;

private:
    Vector(E *head, int length) : head(head), length(length) {}

public:
    static Vector<E> of(int length)
    {
        E *head;
        starpu_malloc((void **)&head, length * sizeof(E));
        return Vector{head, length};
    }

    static Vector<E> from(void *descr)
    {

        E *head = (E *)(STARPU_VECTOR_GET_PTR(descr));
        int length = (int)STARPU_VECTOR_GET_NX(descr);
        return Vector{head, length};
    }

    void fill(E value)
    {
        for (int i = 0; i < length; i++)
        {
            head[i] = value;
        }
    }

    void print()
    {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < length; ++i)
        {
            oss << head[i] << (i == length - 1 ? "" : ", ");
        }
        oss << "]\n";

        std::cout << oss.str();
    };

    void free()
    {
        starpu_free_noflag(head, length * sizeof(E));
    }
};
