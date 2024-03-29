#include "../matrix.hpp"
#include "exact.hpp"

//Разложние Холецкого матрицы A через матрицу U
//А - симметричная положительно определённая матрица
//U - верхнетреугольная матрица, такая что A = U^T * U
static Matrix split(const Matrix &a) {
    int n = a.num_rows();
    Matrix u(n, n);

    u(0, 0) = sqrt(a(0, 0));
    for (int i = 1; i < n; ++i)
        u(0, i) = a(0, i) / u(0, 0);

    for (int k = 1; k < n; ++k) {
        double t = 0;
        for (int i = 0; i < k; ++i)
            t += u(i, k) * u(i, k);
        u(k, k) = sqrt(a(k, k) - t);

        for (int i = k + 1; i < n; ++i) {
            t = 0;
            for (int j = 0; j < k; ++j)
                t += u(j, k) * u(j, i);
            u(k, i) = (a(k, i) - t) / u(k, k);
        }
    }

    return u;
}


//Решить СЛУ методом квадратного корня, U - матрица, полученная разложением split
static void sqrt_method_impl(Matrix &u, std::vector<double> &b) {
    //Ax = b, A = U^T * U => U^T * U * x = b
    //Теперь последовательно решаем две простые СЛУ:
    //1. U^T * y = b
    //2. Ux = y
    
    int n = b.size();
    Transposed<Matrix> uT(u);

    //Решаем U^T * y = b, где U^T - нижнетреугольная матрица
    for (int i = 0; i < n; ++i) {
        b[i] /= uT(i, i);
        for (int j = i + 1; j < n; ++j)
            b[j] -= uT(j, i) * b[i];
    }

    //В b записано решение
#ifdef DEBUG_PRINT
    printf("[ ");
    for (double &b_i: b)
        printf("%5.2f ", b_i);
    printf("]\n");
#endif

    //Решаем U * x = y, где U - верхнетреугольная матрица
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= u(i, i);
        for (int j = i - 1; j >= 0; --j)
            b[j] -= u(j, i) * b[i];
    }

    //В b записано решение изначальной СЛУ
#ifdef DEBUG_PRINT
    printf("[ ");
    for (double &b_i: b)
        printf("%5.2f ", b_i);
    printf("]\n");
#endif
}

void sqrt_method(Matrix &a, Vec &b, Vec &x) {
    Matrix u = split(a);
    sqrt_method_impl(u, b);
    x = b;
}

