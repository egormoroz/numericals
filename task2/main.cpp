#include "exact/exact.hpp"
#include "iterative/iterative.hpp"
#include "matrix.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

using std::cout;
using std::endl;
using std::string;
using std::setprecision;

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        cout << "usage: task2 <method> <path_to_data>" << endl;
        return 1;
    }

    const double EPS = 1e-4;

    string method = argv[1];
    std::ifstream fin(argv[2]);
    if (!fin.is_open()) {
        cout << "failed to open file " << argv[2] << endl;
        return 1;
    }

    Matrix a = mat_from_stream(fin);
    Vec b(a.num_cols()), x(a.num_cols());
    for (auto &i: b)
        fin >> i;


    int its = -1;
    if (method == "gauss")
        gauss(a, b, x);
    else if (method == "sqrt")
        sqrt_method(a, b, x);
    else if (method == "reflection")
        reflection(a, b, x);
    else if (method == "simple")
        its = simple(a, b, x, EPS);
    else if (method == "seidel")
        its = seidel(a, b, x, EPS);
    else if (method == "jacobi")
        its = jacobi(a, b, x, EPS);
    else {
        cout << "unknown method; available methods are: \n"
            << "exact: gauss, sqrt, reflection\n"
            << "iterative: simple, seidel, jacobi" << endl;
        return 1;
    }

    if (its > 0)
        cout << "iterations: " << its << "\n";
    
    cout << "[ ";
    for (auto &i: x)
        cout << setprecision(4) << i << " ";
    cout << "]" << endl;

    return 0;
}

