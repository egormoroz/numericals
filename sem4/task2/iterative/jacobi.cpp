#include "../matrix.hpp"
#include "iterative.hpp"
#include <cstdio>


static double dist(const std::vector<double> &u, 
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

int jacobi(Matrix &a, std::vector<double> &b, 
        std::vector<double> &x, double eps) 
{
    int n = a.num_cols();
    for (int i = 0; i < n; ++i)
        b[i] /= a(i, i);

    for (int i = 0; i < n; ++i) {
        double a_ii = a(i, i);
        for (int j = 0; j < n; ++j)
            a(i, j) = -(!(i == j) * a(i, j) / a_ii);
    }

    std::vector<double> xp = b;
    x = b;
    int its = 0;

    do {
        ++its;
        xp = x;
        mul_mat_vec(a, xp, x);
        for (int i = 0; i < n; ++i)
            x[i] += b[i];
#ifdef DEBUG_PRINT
        printf("[ ");
        for (auto &i: x)
            printf("%.4f ", i);
        printf("]\n");
#endif
    } while (dist(x, xp) > eps && its < MAX_ITERATIONS);

    return its;
}

