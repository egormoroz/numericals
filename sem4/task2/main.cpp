#include "exact/exact.hpp"
#include "iterative/iterative.hpp"
#include "matrix.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>

using std::cout;
using std::endl;
using std::string;
using std::setprecision;
using std::fixed;


//TODO: use arena allocator for matrices and vectors
//because we allocate them in bulk and then drop 
//all the data at once



Matrix gen_ill_matrix(int m, double eps, int N) {
    double val = eps * N;

    Matrix a(m, m);
    for (int i = 0; i < m; ++i) {
        a(i, i) = 1 + val;
        for (int j = i + 1; j < m; ++j)
            a(i, j) = -1 - val;
        for (int j = 0; j < i; ++j)
            a(i, j) = val;
    }

    return a;
}

template<typename F>
void test_ill_conditioned(F &solve, int N, double eps,
        int start_size, int end_size)
{
    for (int m = start_size; m <= end_size; ++m) {
        Matrix a = gen_ill_matrix(m, eps, N);
        std::vector<double> b(m, -1), x(m);
        b[m - 1] = 1;

        int its = solve(a, b, x);
        cout << m;
        if (its > 0)
            cout << ", " << its << " iterations";
        cout << ": [ ";
        for (auto &i: x)
            cout << setprecision(4) << fixed << i << " ";
        cout << "]" << endl;
    }
}

const double EPS = 1e-5;

std::function<int(Matrix&, Vec&, Vec&)> choose_method(
        const char *method_name)
{
    std::function<int(Matrix&, Vec&, Vec&)> f;
    std::string method(method_name);

    auto wrap_exact = [](auto &f) {
        return [f](Matrix &a, Vec &b, Vec &x) {
            f(a, b, x);
            return -1;
        };
    };

    auto wrap_iter = [](auto &f) {
        return [f](Matrix &a, Vec &b, Vec &x) {
            return f(a, b, x, EPS);
        };
    };

    if (method == "gauss")
        f = wrap_exact(gauss);
    else if (method == "sqrt")
        f = wrap_exact(sqrt_method);
    else if (method == "reflection")
        f = wrap_exact(reflection);
    else if (method == "simple")
        f = wrap_iter(simple);
    else if (method == "seidel")
        f = wrap_iter(seidel);
    else if (method == "jacobi")
        f = wrap_iter(jacobi);

    return f;
}

int solve(const char *method_name, const char *file_path) {
    string method(method_name);
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        cout << "failed to open file " << file_path << endl;
        return 1;
    }

    Matrix a = mat_from_stream(fin);
    Vec b(a.num_cols()), x(a.num_cols());
    for (auto &i: b)
        fin >> i;


    auto f = choose_method(method_name);
    if (!f) {
        cout << "unknown method; available methods are: \n"
            << "exact: gauss, sqrt, reflection\n"
            << "iterative: simple, seidel, jacobi" << endl;
        return 1;
    }
    int its = f(a, b, x);

    if (its > 0)
        cout << "iterations: " << its << "\n";
    
    cout << "[ ";
    for (auto &i: x)
        cout << setprecision(4) << fixed << i << " ";
    cout << "]" << endl;

    return 0;
}

int test(const char *method_name, int N, double eps, 
        int start_size, int end_size)
{
    auto f = choose_method(method_name);
    if (!f) {
        cout << "unknown method; available methods are: \n"
            << "exact: gauss, sqrt, reflection\n"
            << "iterative: simple, seidel, jacobi" << endl;
        return 1;
    }

    test_ill_conditioned(f, N, eps, start_size, end_size);
    return 0;
}


int main(int argc, const char* argv[]) {
    if (argc < 2) {
        cout << "usage: task2 <mode> <args>" << endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "solve") {
        if (argc != 4) {
            cout << "invalid solve arguments\nexpected input: "
                << "task2 solve <method> <path_to_data>" << endl;
            return 1;
        }
        return solve(argv[2], argv[3]);
    } else if (mode == "test") {
        const char *invalid_notif = "invalid test arguments\n"
            "expected input: task2 test <method> <N> <epsilon> "
            "<start_size> <end_size>";
        if (argc != 7) {
            cout << invalid_notif << endl;
            return 1;
        }

        try {
            const char *method_name = argv[2];
            int N = std::stoi(argv[3]);
            double eps = std::stod(argv[4]);
            int start_size = std::stoi(argv[5]),
                end_size = std::stoi(argv[6]);

            return test(method_name, N, eps, start_size, end_size);
        } catch (const std::invalid_argument& ex) {
            cout << "invalid argument: " << ex.what() << "\n"
                << invalid_notif << endl;
            return 1;
        }
    } else {
        cout << "unknown mode; available modes are: solve, test" << endl;
        return 1;
    }
}

