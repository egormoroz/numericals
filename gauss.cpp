#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;

//A | B
struct Matrix {
    vector<double*> rows;
    vector<double> data;
    int num_rows, num_cols;

    Matrix(int n_rows, int n_cols) {
        num_cols = n_cols;
        num_rows = n_rows;
        data.resize(n_rows * n_cols);
        rows.resize(n_rows);
        for (int i = 0; i < n_rows; ++i)
            rows[i] = &data[i * n_cols];
    }

    void swap_rows(int i, int j) {
        std::swap(rows[i], rows[j]);
    }

    //[add_to] += k * [add_what]
    void mul_add_rows(int add_to, int add_what, double k) {
        double *to_row = rows[add_to],
            *what_row = rows[add_what];
        for (int i = 0; i < num_cols; ++i)
            to_row[i] += k * what_row[i];
    }

    //[rdx] *= k
    void mul_row(int rdx, double k) {
        double *row = rows[rdx];
        for (int i = 0; i < num_cols; ++i)
            row[i] *= k;
    }

    void print() {
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j)
                printf("%5.2f ", rows[i][j]);
            printf("\n");
        }
        printf("\n");
    }
};

//m: A | B
void gauss(Matrix &m) {
    const double EPS = 1e-5;
    //прямой ход Гаусса
    for (int i = 0; i < m.num_rows - 1; ++i) {
        double a_ii = m.rows[i][i];

        //если очередной ведущий элемент 0, то надо поменять уравнения местами
        //так, чтобы он стал ненулевым (здесь мы заодно ищем уравнение с наибольшим по модулю коэфом)
        if (abs(a_ii) < EPS) {
            double a_ki = a_ii;
            int k = i;
            for (int j = i + 1; j < m.num_rows; ++j) {
                double a_ji = m.rows[j][i];
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
        m.print();

        m.mul_row(i, 1 / a_ii);
        for (int j = i + 1; j < m.num_rows; ++j) {
            double a_ji = m.rows[j][i];
            //не обязательно вычитать весь ряд, т.к. там уже есть нулевые элементы
            //но так проще)
            m.mul_add_rows(j, i, -a_ji);
        }
        m.print();
    }

    double a_nn = m.rows[m.num_rows - 1][m.num_cols - 2];
    if (abs(a_nn) < EPS) {
        printf("matrix is singular, aborting...\n");
        return;
    }

    //Обратный ход Гаусса
    m.mul_row(m.num_rows - 1, 1 / a_nn);
    for (int i = m.num_rows - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            double a_ji = m.rows[j][i];
            //аналогично, здесь изменяются только два стоблца - i-й и последний
            m.mul_add_rows(j, i, -a_ji);
        }
        m.print();
    }
}

int main() {
    freopen("2.txt", "r", stdin);
    int n;
    cin >> n;

    Matrix m(n, n + 1);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n + 1; ++j)
            cin >> m.rows[i][j];

    gauss(m);
}

