#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <cstdio>
#include <algorithm>

FILE* fpt_dump;

void dump_array(const double* a, int n) {
    for (int i = 0; i < n; ++i)
        fprintf(fpt_dump, "%f ", a[i]);
    fprintf(fpt_dump, "\n");
}

double ipoly_lagrange(
        const double* xs,
        const double* ys,
        int n,
        double x)
{
    double s = 0;

    for (int i = 0; i < n; ++i) {
        double num = 1, denum = 1;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            num *= x - xs[j];
            denum *= xs[i] - xs[j];
        }

        s += num / denum * ys[i];
    }

    return s;
}

double ipoly_newton(
        const double* xs,
        const double* ys,
        int n,
        double x)
{
    double y = 0;

    for (int i = 0; i < n; ++i) {
        double omega = 1;
        for (int j = 0; j < i; ++j)
            omega *= x - xs[j];

        double dd = 0;
        for (int j = 0; j <= i; ++j) {
            double denum = 1;
            for (int k = 0; k <= i; ++k) {
                if (j == k) continue;
                denum *= xs[j] - xs[k];
            }
            dd += ys[j] / denum;
        }

        y += dd * omega;
    }

    return y;
}

void linspace(double a, double b, double* xs, int n) {
    if (n == 1) {
        xs[0] = a;
        return;
    }

    const double h = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i)
        xs[i] = a + h * i;
}

void alterspace(double a, double b, double* xs, int n) {
    if (n == 1) {
        xs[0] = a;
        return;
    }
    
    for (int i = 0; i < n; ++i)
        xs[i] = 0.5 
            * ((b - a) 
            * cos(M_PI * (2 * i + 1) / (2 * n))
            + (b + a));
}

double max_diff(const double* u, const double* v, int n) {
    double max_d = fabs(u[0] - v[0]);
    for (int i = 1; i < n; ++i)
        max_d = std::max(max_d, fabs(u[i] - v[i]));
    return max_d;
}

template<typename F, typename InterPoly>
void run(double a, double b, F&& f, InterPoly&& ipoly) {
    constexpr int N = 50;
    constexpr int M = 1000;

    double x_table[N], y_table[N], 
           x_test[M], y_test[M], ip_test[M];

    linspace(a, b, x_test, M);
    for (int i = 0; i < M; ++i)
        y_test[i] = f(x_test[i]);

    dump_array(x_test, M);
    dump_array(y_test, M);

    for (int n = 1; n < N; ++n) {
        linspace(a, b, x_table, n);
        for (int i = 0; i < n; ++i)
            y_table[i] = f(x_table[i]);

        for (int i = 0; i < M; ++i)
            ip_test[i] = ipoly(x_table, y_table, n, x_test[i]);
        printf("%02d %d %.4e ", n, M, max_diff(ip_test, y_test, M));
        dump_array(ip_test, M);

        alterspace(a, b, x_table, n);
        for (int i = 0; i < n; ++i)
            y_table[i] = f(x_table[i]);

        for (int i = 0; i < M; ++i)
            ip_test[i] = ipoly(x_table, y_table, n, x_test[i]);
        printf("%.4e\n", max_diff(ip_test, y_test, M));
        dump_array(ip_test, M);
    }

}

int main() {
    constexpr double a = 1, b = 2;
    auto f = [](double x) { return x + log(x/5); };
    /* constexpr double a = -1, b = 1; */
    /* auto f = [](double x) { return 3 * x - cos(x) - 1; }; */

    fpt_dump = fopen("lagr.txt", "w");
    printf("---------lagrange-------\n");
    run(a, b, f, ipoly_lagrange);
    fclose(fpt_dump);

    fpt_dump = fopen("newt.txt", "w");
    printf("----------newton--------\n");
    run(a, b, f, ipoly_newton);
    fclose(fpt_dump);
}

