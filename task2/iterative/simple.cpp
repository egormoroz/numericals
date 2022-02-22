#include "../matrix.hpp"
#include <fstream>

const int MAX_ITERATIONS = 1000;

// ||A||_inf
double norm(const Matrix &a) {
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
double dist(const std::vector<double> &u, 
        const std::vector<double> &v) 
{
    double n = 0;
    for (int i = 0; i < (int)u.size(); ++i) {
        double t = abs(u[i] - v[i]);
        if (t > n) n = t;
    }
    return n;
}

int solve(Matrix &a, std::vector<double> &b, double eps) {
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

int main(int argc, const char *argv[]) {
    const char *file = "simple.txt";
    if (argc == 2)
        file = argv[1];

    std::ifstream fin(file);
    Matrix a = mat_from_stream(fin);

    std::vector<double> b(a.num_cols());
    for (auto &i: b)
        fin >> i;
    print_mat_extended(a, b);

    printf("%d iterations: [ ", solve(a, b, 1e-5));
    for (auto &i: b)
        printf("%.4f ", i);
    printf("]\n");
}

