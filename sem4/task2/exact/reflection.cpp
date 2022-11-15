#include "../matrix.hpp"
#include "exact.hpp"

//вычисляет матрицу v * v^T и умножает каждый полученный элемент на k
//затем записывает результат в правый нижний угол матрицы mm
static void mul_outer_mult(const std::vector<double> &v, Matrix &mm, double k) {
    int n = v.size(), off = mm.num_cols() - n;
    SubMatrix<Matrix> m(mm, off, off);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m(i, j) = v[i] * v[j] * k;
}

//добавляет к матрице m единичную матрицу
static void add_identity(Matrix &m) {
    int n = m.num_cols();
    for (int i = 0; i < n; ++i)
        m(i, i) += 1;
}

static void reflection_impl(Matrix &a, std::vector<double> &b) {
    const double EPS = 1e-10;

    int n = a.num_cols();
    Matrix u(n, n), aa(n, n);

    std::vector<double> v;
    v.reserve(n);

    for (int k = 0; k < n - 1; ++k) {
        //y_k = [a_kk, ..., a_nk]
        double s = 0;
        v.clear();
        for (int i = k; i < n; ++i) {
            v.push_back(a(i, k));
            s += a(i, k) * a(i, k);
        }

        //v_k = y_k - ||y_k|| * e_k
        double old_v0 = v[0];
        v[0] -= sqrt(s);
        if (abs(v[0]) < EPS) {
            //y_k был коллиниарен e_k (т.е. ниже все нули)
            continue;
        }
        //квадрат нормы вектора v (w = v / ||v||)
        s += -old_v0 * old_v0 + v[0] * v[0];

        //считаем матрицу (-2 / ||v||^2) * v * v^T
        u.set_zeros();
        mul_outer_mult(v, u, -2 / s);

        //получаем итоговую блочную матрицу U^
        add_identity(u);

        //aa = U * A
        dot(u, a, aa);
        //v = U * b
        mul_mat_vec(u, b, v);

        std::swap(a, aa);
        std::swap(b, v);
#ifdef DEBUG_PRINT
        print_mat_extended(a, b);
#endif
    }


    //решаем СЛАУ Ax=b с верхнетреугольной матрицей
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= a(i, i);
        for (int j = i - 1; j >= 0; --j)
            b[j] -= a(j, i) * b[i];
    }

    //В b записано решение изначальной СЛУ
#ifdef DEBUG_PRINT
    printf("[ ");
    for (double &b_i: b)
        printf("%5.2f ", b_i);
    printf("]\n");
#endif
}

void reflection(Matrix &a, Vec &b, Vec &x) {
    reflection_impl(a, b);
    x = b;
}

