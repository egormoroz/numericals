#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

//#define DEBUG_PRINT

#include <vector>

class Matrix;

using Vec = std::vector<double>;

const int MAX_ITERATIONS = 10000;

int simple(Matrix &a, Vec &b, Vec &x, double eps);
int seidel(Matrix &a, Vec &b, Vec &x, double eps);
int jacobi(Matrix &a, Vec &b, Vec &x, double eps);

#endif
