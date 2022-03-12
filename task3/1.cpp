#define _CRT_NO_SECURE_WARNINGS
#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <vector>

constexpr double BND_EPS = 1e-8;
constexpr double LEFT = -0.8 + BND_EPS;
constexpr double RIGHT = 1.2 - BND_EPS;
constexpr int N = 100;
constexpr double EPS = 1e-4;


//x^2 = arcsin(x - 0.2)
//=> f(x) = x^2 - arcsin(x-0.2)
double f(double x) {
    return x*x - asin(x - 0.2);
}

//f'(x) = 2x - 1/sqrt(1-(x-0.2)^2)
double der(double x) {
    return 2. * x - 1. / sqrt(1. - (x - 0.2) * (x - 0.2));
}

void find_initial_guess(int n, std::vector<double> &guesses) {
    double h = (RIGHT - LEFT) / n;

    double xp = LEFT, yp = f(xp);
    for (int i = 1; i <= n; ++i) {
        double x = LEFT + h * i, y = f(x);
        if (y * yp <= 0)
            guesses.push_back((xp + x) / 2);

        xp = x; yp = y;
    }
}

double solve(double x0, double eps) {
    double next = x0, prev;
    do {
        prev = next;
        next = prev - f(prev) / der(prev);
    } while (abs(next - prev) >= eps);

    return next;
}

int main() {
    std::vector<double> guesses;
    find_initial_guess(N, guesses);

    printf("initial guesses:\n");
    for (double i: guesses)
        printf("f(%f) = %f\n", i, f(i));

    printf("roots:\n");
    for (double i: guesses) {
        double root = solve(i, EPS);
        printf("f(%f) = %.2e\n", root, f(root));
    }
}

