#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>

double ipoly_lagrange(double x, 
        const double* xs,
        const double* ys,
        int n)
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

double norm(const double* u, const double* v, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        double t = u[i] - v[i];
        s += t * t;
    }

    return sqrt(s);
}

template<typename F>
void run(double a, double b, F&& f) {
    constexpr int N = 100;
    constexpr int M = 1000;

    double xs[N], ys[N], 
           ts[M], us[M], vs[M];

    linspace(a, b, ts, M);
    for (int i = 0; i < M; ++i)
        us[i] = f(ts[i]);

    for (int n = 1; n < N; ++n) {
        linspace(a, b, xs, n);
        for (int i = 0; i < n; ++i)
            ys[i] = f(xs[i]);

        for (int i = 0; i < M; ++i)
            vs[i] = ipoly_lagrange(ts[i], xs, ys, n);
        printf("%02d %.4e ", n, norm(us, vs, M));

        alterspace(a, b, xs, n);
        for (int i = 0; i < n; ++i)
            ys[i] = f(xs[i]);

        for (int i = 0; i < M; ++i)
            vs[i] = ipoly_lagrange(ts[i], xs, ys, n);
        printf("%.4e\n", norm(us, vs, M));
    }

}

int main() {
    constexpr double a = 1, b = 2;
    auto f = [](double x) { return x + log(x/5); };
    /* constexpr double a = -1, b = 1; */
    /* auto f = [](double x) { return 3 * x - cos(x) - 1; }; */

    auto g = [f](double x) { return fabs(x) * f(x); };

    printf("---------f(x)-------\n");
    run(a, b, f);
    printf("------|x|f(x)-------\n");
    run(a, b, g);
}

