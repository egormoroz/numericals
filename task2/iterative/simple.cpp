#include "../matrix.hpp"
#include "iterative.hpp"


// ||A||_inf
static double norm(const Matrix &a) {
    double s = 0;
    for (int i = 0; i < a.num_rows(); ++i) {
        double t = 0;
        for (int j = 0; j < a.num_cols(); ++j)
            t += abs(a(i, j));
        if (t > s) s = t;
    }

    return s;
}

//||v||_inf
static double dist(const std::vector<double> &u, 
        const std::vector<double> &v) 
{
    double n = 0;
    for (int i = 0; i < (int)u.size(); ++i) {
        double t = abs(u[i] - v[i]);
        if (t > n) n = t;
    }
    return n;
}

int simple_impl(Matrix &a, std::vector<double> &b, double eps) {
    int n = a.num_cols();
    Transposed<Matrix> aT(a);

    Matrix aa(n, n);
    dot(aT, a, aa);

    std::vector<double> bb(n);
    mul_mat_vec(aT, b, bb);

    double nu = 1 / norm(aa);
    for (auto &i: bb)
        i *= nu;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aa(i, j) *= -nu;
            aa(i, j) += (i == j);
        }
    }

    std::vector<double> prev = bb, &next = b,
        &c = bb;
    Matrix &B = aa;

    int its = 0;
    next = bb;
    do {
        prev = next;
        mul_mat_vec(B, prev, next);
        for (int i = 0; i < n; ++i)
            next[i] += c[i];
    } while (dist(next, prev) > eps && ++its < MAX_ITERATIONS);

    return its;
}

int simple(Matrix &a, Vec &b, Vec &x, double eps) {
    x = b;
    return simple_impl(a, x, eps);
}

