#pragma once

#include "headers_app.hpp"

template <typename E>
void gemm_seq(Matrix<E> &A, Matrix<E> &B, Matrix<E> &C, Coef<E> coef)
{
    int m = C.matrix_height();
    int n = C.matrix_width();
    int k = A.matrix_width();

    int ldA = A.matrix_height();
    int ldB = B.matrix_height();
    int ldC = C.matrix_height();

    E *headA = A.head();
    E *headB = B.head();
    E *headC = C.head();

    blas<E>::gemm('N', 'N',
                  m, n, k,
                  coef.alpha,
                  headA, ldA,
                  headB, ldB,
                  coef.beta,
                  headC, ldC);
}