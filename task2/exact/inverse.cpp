#include "../matrix.hpp"
#include <fstream>

double vdot(const std::vector<double> &u, 
        const std::vector<double> &v) 
{
    double s = 0;
    for (int i = 0; i < (int)u.size(); ++i)
        s += u[i] * v[i];

    return s;
}

Matrix next_inverse(Matrix &a, Matrix &prev_inv) {
    int n = a.num_cols();
    Matrix inv(n, n);

    //o_O
    //можно уменьшить кол-во векторов, одновременно удерживаемых в памяти
    std::vector<double> v(n - 1), 
        u(n - 1), c(n - 1), w(n - 1), s(n - 1);

    for (int i = 0; i < n - 1; ++i) {
        v[i] = a(i, n - 1);
        u[i] = a(n - 1, i);
    }

    //c = A_{n-1}^(-1) * v
    mul_mat_vec(prev_inv, v, c);

    //alpha = 1 / (a_n - u^T * C)
    double alpha = 1 / (a(n - 1, n - 1) - vdot(u, c));

    //s^T = -alpha * u^T * A_{n-1}^(-1)
    mul_vec_mat(u, prev_inv, s);
    for (auto &i: s)
        i *= -alpha;

    for (int i = 0; i < n - 1; ++i)
        w[i] = -alpha * c[i];
    
    
    //B_{n-1} = A_{n-1}^(-1) - c * s^T
    for (int i = 0; i < n - 1; ++i)
        for (int j = 0; j < n - 1; ++j)
            inv(i, j) = prev_inv(i, j) - c[i] * s[j];

    for (int i = 0; i < n - 1; ++i)
        inv(i, n - 1) = w[i];
    for (int i = 0; i < n - 1; ++i)
        inv(n - 1, i) = s[i];
    inv(n - 1, n - 1) = alpha;

    return inv;
}

int main() {
    std::ifstream fin("inv.txt");
    Matrix a = mat_from_stream(fin);
    Matrix prev_inv = mat_from_stream(fin);
    int n = a.num_cols();

    print_mat(a);
//    print_mat(prev_inv);

    Matrix inv = next_inverse(a, prev_inv);
    print_mat(inv);

    Matrix id(n, n);
    dot(a, inv, id);
    print_mat(id);
}

