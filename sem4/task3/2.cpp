#define _CRT_NO_SECURE_WARNINGS
#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <vector>

struct Vec2 {
    double x, y;

    double norm_sqr() const { return x * x + y * y; }

    Vec2 operator-(const Vec2 &u) const {
        return Vec2{ x - u.x, y - u.y };
    }

    Vec2 operator+(const Vec2 &u) const {
        return Vec2{ x + u.x, y + u.y };
    }

    Vec2 operator-() const {
        return Vec2{ -x, -y };
    }
};

struct Mat22 {
    Mat22(double a00, double a01, double a10, double a11)
        : a00(a00), a01(a01), a10(a10), a11(a11) {}

    Mat22(Vec2 col0, Vec2 col1)
        : col0(col0), col1(col1) {}

    double det() const {
        return a00 * a11 - a10 * a01;
    }

    union {
        struct {
            double a00, a10, a01, a11;
        };
        struct {
            Vec2 col0, col1;
        };
    };
};

/*
 * sin(x + 0.5) - y - 1 = 0
 * x + cos(y - 2) - 2 = 0
 * */
Vec2 f(Vec2 v) {
    return Vec2 {
        sin(v.y + 2.0) - v.x - 1.5,
        v.y + cos(v.x - 2.0) - 0.5,
        /* sin(v.x + 0.5) - v.y - 1, */
        /* v.x + cos(v.y - 2) - 2 */
    };
}

/*
 * [[ cos(x + 0.5),         -1 ],
 *  [            1, -sin(y - 2)]]
 * */
Mat22 jacobi(Vec2 v) {
    return Mat22 {
        -1, cos(v.y + 2),
        -sin(v.x - 2), 1
        /* cos(v.x + 0.5), -1, 1, -sin(v.y - 2) */
    };
}

Vec2 solve_leq(Mat22 A, Vec2 b) {
    double d = A.det();
    return Vec2 {
        Mat22(b, A.col1).det() / d,
        Mat22(A.col0, b).det() / d
    };
}

Vec2 solve(Vec2 x0, double eps) {
    double eps_sqr = eps * eps;
    Vec2 dx;

    do {
        dx = solve_leq(jacobi(x0), -f(x0));
        x0 = x0 + dx;
        printf("[%5f, %5f]\n", x0.x, x0.y);
    } while (dx.norm_sqr() >= eps_sqr);

    return x0;
}

int main() {
    Vec2 root = solve({-1.7, 1.3}, 1e-5),
         val = f(root);
    printf("F(%f, %f) = [ %.2e, %.2e ]\n",
            root.x, root.y, val.x, val.y);
}

