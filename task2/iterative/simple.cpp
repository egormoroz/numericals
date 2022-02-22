#include "../matrix.hpp"
#include <fstream>

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
double norm(const std::vector<double> &v) {
    double n = 0;
    for (auto &i: v)
        if (abs(i) > n)
            n = abs(i);
    return n;
}

void solve(Matrix &a, std::vector<double> &b, int max_iterations) {
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

    for (int its = 0; its < max_iterations; ++its) {
        mul_mat_vec(B, prev, next);
        for (int i = 0; i < n; ++i)
            next[i] += c[i];
        prev = next;
    }
}

int main() {
    std::ifstream fin("simple.txt");
    Matrix a = mat_from_stream(fin);

    std::vector<double> b(a.num_cols());
    for (auto &i: b)
        fin >> i;

    solve(a, b, 1000);

    printf("[ ");
    for (auto &i: b)
        printf("%5.2f ", i);
    printf("]\n");
}

