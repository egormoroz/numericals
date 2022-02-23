#include "../matrix.hpp"
#include "exact.hpp"

static void gauss_impl(Matrix &origm) {
    RowsRemapped<Matrix> m(origm);
    using Mat = decltype(m);

    const double EPS = 1e-10;
    //прямой ход Гаусса
    for (int i = 0; i < m.num_rows() - 1; ++i) {
        double a_ii = m(i, i);

        //если очередной ведущий элемент 0, то надо поменять уравнения местами
        //так, чтобы он стал ненулевым (здесь мы заодно ищем уравнение с наибольшим по модулю коэфом)
        if (abs(a_ii) < EPS) {
            double a_ki = a_ii;
            int k = i;
            for (int j = i + 1; j < m.num_rows(); ++j) {
                double a_ji = m(j, i);
                if (abs(a_ji) > abs(a_ki)) {
                    a_ki = a_ji;
                    k = j;
                }
            }

            if (abs(a_ki) < EPS) {
                //не смогли найти ненулевой ведущий элемент - матрица А вырождена
                printf("matrix is singular, aborting...\n");
                return;
            }

            m.swap_rows(i, k);
            a_ii = a_ki;
        }

        Row<Mat> a_i(m, i);
        a_i /= a_ii;
#ifdef DEBUG_PRINT
        print_mat(m);
#endif

        for (int j = i + 1; j < m.num_rows(); ++j) {
            Row<Mat> a_j(m, j);
            //не обязательно вычитать всю строку, т.к. там уже есть нулевые элементы
            //но так проще)
            a_j.mul_add(a_i, -m(j, i));
        }
#ifdef DEBUG_PRINT
        print_mat(m);
#endif
    }

    double a_nn = m(m.num_rows() - 1, m.num_rows() - 1);
    if (abs(a_nn) < EPS) {
        printf("matrix is singular, aborting...\n");
        return;
    }
    Row<Mat> last(m, m.num_rows() - 1);
    last /= a_nn;

#ifdef DEBUG_PRINT
    print_mat(m);
#endif

    //Обратный ход Гаусса
    for (int i = m.num_rows() - 1; i > 0; --i) {
        double b_i = m(i, m.num_cols() - 1);
        for (int j = i - 1; j >= 0; --j) {
            m(j, m.num_cols() - 1) -= b_i * m(j, i);
            m(j, i) = 0;
        }
#ifdef DEBUG_PRINT
        print_mat(m);
#endif
    }
}

void gauss(Matrix &a, Vec &b, Vec &x) {
    int n = a.num_cols();
    Matrix ab(n, n + 1);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            ab(i, j) = a(i, j);
    for (int i = 0; i < n; ++i)
        ab(i, n) = b[i];

    gauss_impl(ab);
    for (int i = 0; i < n; ++i)
        x[i] = ab(i, n);
}

