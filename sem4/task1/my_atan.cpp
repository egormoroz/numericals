#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstdint>

//polynomial approximation of sqrt(x) on [1; 1.5]
double sqrt_poly_approx(double x) {
    const double A = -0.1086325;
    const double B = 0.71608019;
    const double C = 0.39256629; 
    return x * (A * x + B) + C;
}

double my_sqrt(double a, double eps) {
    double next = sqrt_poly_approx(a), cur;

    do {
        cur = next;
        next = (cur + a / cur) / 2;
    } while (abs(next - cur) >= eps);

    return next;
}

template<typename F>
double simpson(F &f, double a, double b, int n) {
    assert(n % 2 == 0);
    double h = (b - a) / n, s1 = 0, s2 = 0;

    for (int i = 1; i < n / 2; ++i)
        s1 += f(a + 2 * i * h);

    for (int i = 1; i <= n / 2; ++i)
        s2 += f(a + (2 * i - 1) * h);

    return h / 3 * (f(a) + 2 * s1 + 4 * s2 + f(b));
}

double my_atan(double x) {
    if (x < 0)
        return -my_atan(-x);
    if (x > 1)
        return M_PI_2 - my_atan(1 / x);

    const int N = 16;
    const double EPS = 1e-8;

    for (int i = 0; i < N; ++i)
        x = x / (1 + my_sqrt(1 + x*x, EPS));

    auto atan_der = [](double x) { return 1 / (1 + x*x); };
    return (1 << N) * simpson(atan_der, 0, x, 2);
}

int main() {
    //prints zero
    printf("%.4e", abs(my_atan(1) - M_PI_4));
}

