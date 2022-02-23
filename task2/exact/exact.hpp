#ifndef EXACT_HPP
#define EXACT_HPP

#include <vector>
struct Matrix;

using Vec = std::vector<double>;

void gauss(Matrix &a, Vec &b, Vec &x);
void sqrt_method(Matrix &a, Vec &b, Vec &x);
void reflection(Matrix &a, Vec &b, Vec &x);

void inverse(Matrix &a, Matrix &prev_inv);

#endif
