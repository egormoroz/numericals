#include <cstdio>
#include <vector>
#include <cassert>

class Matrix {
public:
    Matrix(int rows = 0, int cols = 0)
        : rows_(rows), cols_(cols),
          data_(rows * cols, 0.) 
    {
        assert(rows >= 0 && cols >= 0);
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    double& operator()(int row, int col) {
        return data_[index(row, col)];
    }

    const double& operator()(int row, int col) const {
        return data_[index(row, col)];
    }

    void pretty_print() {
        printf("[");
        for (int row = 0; row < rows_; ++row) {
            if (row != 0)
                printf(" ");
            printf("[ ");
            for (int col = 0; col < cols_; ++col)
                printf("%6f ", data_[index(row, col)]);

            printf("]");
            if (row != rows_ - 1)
                printf("\n");
        }
        printf("]\n");
    }

private:
    size_t index(int row, int col) const {
        assert(row >= 0 && col >= 0 
            && row < rows_ && col < cols_);
        return static_cast<size_t>(row * cols_ + col);
    }

    int rows_, cols_;
    std::vector<double> data_;
};

//добавить к строке to строку from, домноженную на k
//позже можно будет добавить адаптеры-обёртки
void add_row(Matrix &m, int to, int from, double k) {
    assert(to != from);
    for (int i = 0; i < m.cols(); ++i)
        m(to, i) += k * m(from, i);
}

//умножить строку row на число k
void mul_row(Matrix &m, int row, double k) {
    for (int i = 0; i < m.cols(); ++i)
        m(row, i) *= k;
}

//легче просто считать, что это уже расширенная матрица A | b
void gauss_simple(Matrix &a) {
    assert(a.rows() + 1 == a.cols());

    //n неизвестных совпадает с кол-вом уравнений(строк)
    int n = a.rows();

    //пока что предполагаем, что не нарвёмся на нули на диагонали
    for (int i = 0; i < n; ++i) {
        //нормируем строку
        mul_row(a, i, 1/a(i, i));

        //обнуляем весь столбец под элементом a_ii
        //за счёт вычета строки A_i из всех последующих,
        //домноженных на соотв. коэф
        for (int j = i + 1; j < n; ++j)
            add_row(a, j, i, -a(j, i));
    }

    //получили верхнетреугольную матрицу, а на диагонали стоят
    //единицы; применяем обратный ход Гаусса
    //т.е. делам то же самое, что и выше, только снизу вверх
    //здесь уже достаточно вычитать лишь два элемент
    //а не всю строку (т.к. все остальные элементы - нули)
    for (int i = n - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            a(j, n) -= a(j, i) * a(i, n);
            //как бы вычли, получив ноль
            a(j, i) = 0;
        }
    }
}

//евклидово расстояние векторов u и v, т.е. ||u - v||
double dist(const std::vector<double> &u, 
        const std::vector<double> &v) 
{
    double s = 0;
    int n = (int)u.size();
    for (int i = 0; i < n; ++i) {
        double t = u[i] - v[i];
        s += t * t;
    }

    return sqrt(s);
}

/* a, b, x - обозначения из записи Ax=b
 * eps - требуемая точность.
 *
 * Все входные данные модифицируются, 
 * в переменную "x" записывается ответ.
 * */
int seidel(Matrix &a, std::vector<double> &b,
        std::vector<double> &x, double eps)
{
    int n = a.cols();

    //1. Делим все строки на a_ii
    //и "переносим" все неизвестные кроме x_i вправо.
    //Другими словам, строим матрицу C и вектор d из методички
    for (int i = 0; i < n; ++i) {
        double a_ii = a(i, i);

        b[i] /= a_ii;

        for (int j = 0; j < n; ++j) {
            a(i, j) /= a_ii;
            if (i != j)
                a(i, j) *= -1;
        }
    }

    //2. Теперь у нас в переменной "a" хранится матрица C
    //и в "b" - вектор d. 

    int num_iterations = 0;
    //предыдущее приближение
    //в качестве начального приближения берём нулевой вектор
    //подставив который получаем x_1 = C * 0 + d = d
    std::vector<double> xp = b;
    x = b;

    do {
        num_iterations++;
        xp = x;

        //считаем согласно формуле из методички x_{k+1}
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                s += a(i, j) * x[j];
            }
            x[i] = s + b[i];
        }

    //прекращаем считать, когда ||x_{k+1} - x_k|| < eps
    //либо превышено количество итераций (чтобы избежать беск. цикл)
    } while (dist(xp, x) >= eps && num_iterations < 1000);

    return num_iterations;
}

void test_gauss() {
    //решение - [ 6 9 ]
    //[[ 2 2 30 ]
    // [ 3 6 72 ]]
    Matrix m(2, 3);
    m(0, 0) = 2;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 6;
    m(0, 2) = 30;
    m(1, 2) = 72;

    m.pretty_print();
    gauss_simple(m);

    m.pretty_print();
}

void test_seidel() {
    //решение - [ 6 9 ]
    //[[ 2 2 30 ]
    // [ 3 6 72 ]]
    Matrix m(2, 2);
    m(0, 0) = 2;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 6;

    std::vector<double> b = { 30.0, 72.0 };
    std::vector<double> x = { 0.0, 0.0 };

    int num_iterations = seidel(m, b, x, 1e-5);
    printf("%d iterations\n", num_iterations);

    printf("[ ");
    for (double x_i: x)
        printf("%6f ", x_i);
    printf("]\n");
}

int main() {
    test_seidel();
}
