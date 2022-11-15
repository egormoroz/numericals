#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <numeric>
#include <cassert>
#include <istream>
#include <cstdio>

struct Matrix;

template<typename T>
struct Transposed;

template<typename T>
struct RowsRemapped;


template<typename Mat = Matrix>
struct Row {
    Row(Mat &m, int row)
        : m_mat(m), m_row(row) {}

    int size() const { return m_mat.num_cols(); }
    int row_num() const { return m_row; }

    template<typename TRow>
    Row& mul_add(const TRow &r, double k) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) += r(col) * k;
        return *this;
    }

    double& operator()(int col) { return m_mat(m_row, col); }
    const double& operator()(int col) const { return m_mat(m_row, col); }

    template<typename TRow>
    Row& operator+=(const TRow &r) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) += r(col);
        return *this;
    }

    template<typename TRow>
    Row& operator-=(const TRow &r) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) -= r(col);
        return *this;
    }

    Row& operator*=(double k) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) *= k;
        return *this;
    }

    Row& operator/=(double k) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) /= k;
        return *this;
    }

    template<typename TRow>
    Row& operator=(const TRow &r) {
        for (int col = 0; col < size(); ++col)
            m_mat(m_row, col) = r(col);
        return *this;
    }

private:
    Mat &m_mat;
    int m_row;
};

template<typename Mat = Matrix>
struct Transposed {
    Transposed(Mat &m)
        : m_mat(m) {}

    int num_rows() const { return m_mat.num_cols(); }
    int num_cols() const { return m_mat.num_rows(); }

    double& operator()(int row, int col) { return m_mat(col, row); }
    const double& operator()(int row, int col) const { return m_mat(col, row); }

private:
    Mat &m_mat;
};

template<typename Mat = Matrix>
struct RowsRemapped {
    RowsRemapped(Mat &m) : m_mat(m), m_indices(m.num_rows()) {
        std::iota(m_indices.begin(), m_indices.end(), 0);
    }

    int num_rows() const { return m_mat.num_rows(); }
    int num_cols() const { return m_mat.num_cols(); }

    void swap_rows(int i, int j) {
        std::swap(m_indices[i], m_indices[j]);
    }

    double& operator()(int row, int col) { 
        return m_mat(m_indices[row], col);
    }

    const double& operator()(int row, int col) const { 
        return m_mat(m_indices[row], col);
    }

private:
    std::vector<int> m_indices;
    Mat &m_mat;
};

template<typename Mat>
struct SubMatrix {
    SubMatrix(Mat &m, int off_row, int off_col)
        : m_mat(m), m_off_row(off_row), m_off_col(off_col) {}

    int num_rows() const { return m_mat.num_rows() - m_off_row; }
    int num_cols() const { return m_mat.num_cols() - m_off_row; }

    double& operator()(int row, int col) {
        return m_mat(row + m_off_row, col + m_off_col);
    }

    const double& operator()(int row, int col) const {
        return m_mat(row + m_off_row, col + m_off_col);
    }

private:
    Mat &m_mat;
    int m_off_row, m_off_col;
};

struct Matrix {
    Matrix(int nrows, int ncols) 
        : m_data(nrows * ncols), m_nrows(nrows), m_ncols(ncols) {}

    template<typename Mat>
    Matrix(const Mat &m) 
        : m_data(m.num_rows() * m.num_cols()), m_nrows(m.num_rows()),
          m_ncols(m.num_cols())
    {
        for (int i = 0; i < m_nrows; ++i)
            for (int j = 0; j < m_ncols; ++j)
                (*this)(i, j) = m(i, j);
    }

    void set_identity() {
        int n = m_nrows;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                (*this)(i, j) = i == j;
    }

    void set_zeros() {
        std::fill(m_data.begin(), m_data.end(), 0);
    }

    int num_rows() const { return m_nrows; }
    int num_cols() const { return m_ncols; }

    double& operator()(int row, int col) { 
        return m_data[row * m_ncols + col];
    }

    const double& operator()(int row, int col) const { 
        return m_data[row * m_ncols + col];
    }

private:
    int m_nrows, m_ncols;
    std::vector<double> m_data;
};

template<typename A, typename B>
void dot(const A &a, const B &b, Matrix &c) {
    assert(a.num_cols() == b.num_rows());
    int rows = a.num_rows(), cols = b.num_cols();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double c_ij = 0;
            for (int k = 0; k < a.num_cols(); ++k)
                c_ij += a(i, k) * b(k, j);
            c(i, j) = c_ij;
        }
    }
}

template<typename Mat>
void mul_mat_vec(const Mat &m, const std::vector<double> &v, 
        std::vector<double> &u) 
{
    int n = v.size();
    u.resize(n);
    for (int i = 0; i < n; ++i) {
        double u_i = 0;
        for (int j = 0; j < n; ++j)
            u_i += m(i, j) * v[j];
        u[i] = u_i;
    }
}

template<typename Mat>
void mul_vec_mat(const std::vector<double> &v, 
        Mat &m, std::vector<double> &u)
{
    int n = v.size();
    u.resize(n);
    for (int i = 0; i < n; ++i) {
        double u_i = 0;
        for (int j = 0; j < n; ++j)
            u_i += m(j, i) * v[j];
        u[i] = u_i;
    }
}

template<typename Mat>
void add(Mat &lhs, const Mat &rhs) {
    int rows = lhs.num_rows(), cols = lhs.num_cols();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            lhs(i, j) += rhs(i, j);
}

template<typename Mat>
void sub(Mat &lhs, const Mat &rhs) {
    int rows = lhs.num_rows(), cols = lhs.num_cols();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            lhs(i, j) -= rhs(i, j);
}

inline Matrix mat_from_stream(std::istream &is) {
    int rows, cols;
    is >> rows >> cols;

    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            is >> m(i, j);

    return m;
}

template<typename Mat>
void print_mat(const Mat &m) {
    for (int i = 0; i < m.num_rows(); ++i) {
        for (int j = 0; j < m.num_cols(); ++j)
            printf("%5.2f ", m(i, j));
        printf("\n");
    }
    printf("\n");
}

template<typename Mat>
void print_mat_extended(const Mat &m, const std::vector<double> &v) {
    for (int i = 0; i < m.num_rows(); ++i) {
        for (int j = 0; j < m.num_cols(); ++j)
            printf("%5.2f ", m(i, j));
        printf("%5.2f\n", v[i]);
    }
    printf("\n");
}

#endif
