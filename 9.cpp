//z(x) = sqrt(sin(x + 0.74)) * sh(0.8x^2 + 0.1)
//x = 0.1(0.01)0.2
//eps = 10^-6

#include <cstdio>
#include <cmath>

using real = double;

//incrementally evaluated factorial
struct Factorial {
    Factorial() = default;
    int n = 1, val = 1;
    
    int operator()(int k) {
        if (!k) return 1;

        for (; n <= k; ++n)
            val *= n;

        return val;
    }
};

//incrementally evaluated power
struct Power {
    Power(real x)
        : x(x), val(1) {}

    int n = 1;
    real x, val;

    real operator()(int k) {
        if (!k) return 1;

        for (; n <= k; ++n)
            val *= x;

        return val;
    }
};

real approx_sin(real x, int n) {
    real s = 0, sign = 1;
    Factorial factorial;
    Power power(x);


    for (int k = 0; k <= n; ++k) {
        s += sign * power(2 * k + 1) / factorial(2 * k + 1);
        sign = -sign;
    }

    return s;
}

real approx_sh(real x, int n) {
    real s = 0;
    Factorial factorial;
    Power power(x);

    for (int k = 0; k <= n; ++k)
        s += power(2 * k + 1) / factorial(2 * k + 1);

    return s;
}

real approx_sqrt(real a, real eps) {
    real next = 1, cur;

    do {
        cur = next;
        next = (cur + a / cur) / 2;
    } while (abs(next - cur) >= eps);

    return next;
}

int main() {
    real a = 0.1, b = 0.2, step = 0.01;
    const real eps1 = 1e-6 / 0.27;
    const real eps2 = 1e-6 / 3;
    const real eps3 = 1e-6 / 3 / 0.14;
    const int n = 4;
    const int p = 2;

    printf("   x |          z*(x) |           z(x) |       diff\n");
    for (real x = a; x <= b + step / 2; x += step) {
        real approx = approx_sqrt(approx_sin(x + 0.74, n), eps3) * approx_sh(0.8 * x * x + 0.1, p);
        real z = sqrt(sin(x + 0.74)) * sinh(0.8 * x * x + 0.1);
        real diff = abs(approx - z);
        printf("%.2f | %.12f | %.12f | %.4e\n", x, approx, z, diff);
    }
}

