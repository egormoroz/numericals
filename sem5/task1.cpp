#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>

using namespace std;

//Реализация вектора из R^3
struct Vec3 {
    Vec3()
        : x(0), y(0), z(0)
    {}

    explicit Vec3(double x, double y, double z)
        : x(x), y(y), z(z) {}

    /*
     * К коомпонентам ветора v можно обращаться лицо как v.x, v.y, v.z
     * или то же самое v.xs[0], v.xs[1], v.xs[2]. Иногда v.xs[k] удобнее.
     * */
    union {
        struct {
            double x, y, z;
        };

        double xs[3];
    };

    //Евклидова норма вектора 
    double norm() const {
        return sqrt(norm_squared());
    }

    //Квадрат евклидовой нормы вектора
    double norm_squared() const {
        return x * x + y * y + z * z;
    }
};

//Квадратная матрица 3х3
class Mat3 {
public:
    Mat3() = default;
    Mat3(double a00, double a01, double a02,
        double a10, double a11, double a12,
        double a20, double a21, double a22)
        : data_{ { a00, a01, a02 },
                 { a10, a11, a12 },
                 { a20, a21, a22 } }
    {}


    //Величина диагонального преобладания
    double diag_dom_magnitude() const {
        double delta = 1e99;
        for (int i = 0; i < 3; ++i) {
            double s = 0;
            for (int j = 0; j < 3; ++j)
                if (i != j)
                    s += fabs(data_[i][j]);

            delta = min(delta, fabs(data_[i][i]) - s);
        }

        return delta;
    };

    //Обратиться к элементу матрицы
    double& operator()(int i, int j) { return data_[i][j]; }
    double operator()(int i, int j) const { return data_[i][j]; }


private:
    double data_[3][3]{};
};

//Произведение матрицы на ветор 
Vec3 operator*(const Mat3& m, const Vec3& v) {
    Vec3 u;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            u.xs[i] += m(i, j) * v.xs[j];

    return u;
}

//Сложение двух векторов
Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

//Разность двух векторов
Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

//Вычесть из вектора другой вектор
Vec3& operator-=(Vec3& u, const Vec3& v) {
    u = u - v;
    return u;
}

//Унарный минус для вектора 
Vec3 operator-(const Vec3& v) {
    return Vec3(-v.x, -v.y, -v.z);
}

//Скалярное произведение векторов
double operator*(const Vec3& u, const Vec3& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

//Умножение вектора на число
Vec3 operator*(const Vec3& v, double k) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}

//Умножение вектора на число
Vec3 operator*(double k, const Vec3& v) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}

//Деление вектора на число
Vec3& operator/=(Vec3& v, double k) {
    v = v * (1 / k);
    return v;
}

//Решить СЛАУ Ax=b методом Гаусса
Vec3 solve(const Mat3& a, Vec3 b) {
    Vec3 rows[3];
    for (int i = 0; i < 3; ++i)
        rows[i] = Vec3(a(i, 0), a(i, 1), a(i, 2));

    auto el = [&](int i, int j) { return rows[i].xs[j]; };

    //Прямой ход
    for (int i = 0; i < 3; ++i) {
        b.xs[i] /= el(i, i);
        rows[i] /= el(i, i);

        for (int j = i + 1; j < 3; ++j) {
            b.xs[j] -= b.xs[i] * el(j, i);
            rows[j] -= rows[i] * el(j, i);
        }
    }

    //Обратный ход
    b.xs[1] -= b.xs[2] * el(1, 2);
    b.xs[0] -= b.xs[2] * el(0, 2);

    b.xs[0] -= b.xs[1] * el(0, 1);

    return b;
}

//Описание квадратичной функции
//F(x) = 1/2 x^T * A * x + bx + c
struct QuadraticFn {
    Mat3 a;
    Vec3 b;
    double c = 0;

    QuadraticFn(int N)
        : a(4, 1, 1,
            1, 6 + 0.2 * N, -1,
            1, -1, 8 + 0.2 * N),
        b(1, -2, 3),
        c(N)
    {
    }

    //Вычислить градиент в точке
    Vec3 grad(const Vec3& x) const {
        return a * x + b;
    }

    //Вычислить функцию в точке
    double operator()(const Vec3& x) const {
        return 0.5 * x * (a * x) + b * x + c;
    }
};

//Допустимая погрешность
const double EPS = 1e-6;

//МНГС
Vec3 mngs(const QuadraticFn& f) {
    //начальная точка (проинициализирована нулём)
    Vec3 x;
    double dm = f.a.diag_dom_magnitude();

    //не более 1000 итераций, чтобы не возникло бск. цикла
    for (int iter = 0; iter < 1000; ++iter) {
        //Отчёт о текущей итерации
        printf("%02d (%.6f, %.6f, %.6f) f(x) = %.6f\n",
            iter, x.x, x.y, x.z, f(x));

        //В МНГС в кач-ве q выступает градиент
        Vec3 q = f.grad(x);
        //Если оценка на абс. погрешность < EPS, то завершаем работу 
        if (q.norm() / dm < EPS)
            break;

        double nu = -q.norm_squared() / (q * (f.a * q));
        x = x + nu * q;
    }

    return x;
}

//МНПС
Vec3 coord(const QuadraticFn& f) {
    double dm = f.a.diag_dom_magnitude();

    //вектор q выбирается из этих ортов
    const Vec3 orts[3] = {
        Vec3(1, 0, 0),
        Vec3(0, 1, 0),
        Vec3(0, 0, 1)
    };

    //начальная точка (проинициализирована нулём)
    Vec3 x;

    //не более 1000 итераций, чтобы не возникло бск. цикла
    for (int iter = 0; iter < 1000; ++iter) {
        //Отчёт о текущей итерации
        printf("%02d (%.6f, %.6f, %.6f) f(x) = %.6f\n",
            iter, x.x, x.y, x.z, f(x));

        //Если оценка на абс. погрешность < EPS, то завершаем работу 
        if ((f.a * x + f.b).norm() / dm < EPS)
            break;

        Vec3 q = orts[iter % 3];
        double nu = -q * (f.a * x + f.b);
        nu /= q * (f.a * q);

        x = x + nu * q;
    }

    return x;
}

void print_mat(const Mat3 &a) {
    printf("[[ %9.6f %9.6f %9.6f ]\n",  a(0, 0), a(0, 1), a(0, 2));
    printf(" [ %9.6f %9.6f %9.6f ]\n",  a(1, 0), a(1, 1), a(1, 2));
    printf(" [ %9.6f %9.6f %9.6f ]]\n", a(2, 0), a(2, 1), a(2, 2));
}

void print_vec(const Vec3 &v) {
    printf("[%9.6f, %9.6f, %9.6f]\n", v.x, v.y, v.z);
}

int main() {
    const int N = 8;
    QuadraticFn f(N);

    Vec3 x = solve(f.a, -f.b);

    double dm = f.a.diag_dom_magnitude();
    printf("%f\n", dm);

    print_mat(f.a);
    print_vec(f.b);

    printf("gradient\n");
    printf("grad absulute error: %e\n", (mngs(f) - x).norm());
    printf("coord\n");
    printf("coord absulute error: %e\n", (coord(f) - x).norm());
}

