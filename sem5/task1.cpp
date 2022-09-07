#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>

using namespace std;

struct Vec3 {
    Vec3()
        : x(0), y(0), z(0) 
    {}

    explicit Vec3(double x, double y, double z)
        : x(x), y(y), z(z) {}

    union {
        struct {
            double x, y, z;
        };

        double xs[3];
    };

    double norm() const {
        return sqrt(norm_squared());
    }

    double norm_squared() const {
        return x * x + y * y + z * z;
    }
};

class Mat3 {
public:
    Mat3() = default;
    Mat3(double a00, double a01, double a02,
         double a10, double a11, double a12,
         double a20, double a21, double a22)
        : data_{ { a00, a01, a02 },
                 { a10, a11, a12 },
                 { a20, a21, a22 } }
    {}


    double diag_dom_magnitude() const {
        double delta = 1e99;
        for (int i = 0; i < 3; ++i) {
            double s = 0;
            for (int j = 0; j < 3; ++j)
                if (i != j)
                    s += fabs(data_[i][j]);

            delta = min(delta, fabs(data_[i][i]) - s);
        }

        return delta;
    };

    double& operator()(int i, int j) { return data_[i][j]; }
    double operator()(int i, int j) const { return data_[i][j]; }


private:
    double data_[3][3]{};
};

Vec3 operator*(const Mat3 &m, const Vec3 &v) {
    Vec3 u;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            u.xs[i] += m(i, j) * v.xs[j];

    return u;
}

Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

Vec3 operator-(const Vec3 &v) {
    return Vec3(-v.x, -v.y, -v.z);
}

double operator*(const Vec3 &u, const Vec3 &v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

Vec3 operator*(const Vec3 &v, double k) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}

Vec3 operator*(double k, const Vec3 &v) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}

//F(x) = 1/2 x^T * A * x + bx + c
struct QuadraticFn {
    Mat3 a;
    Vec3 b;
    double c = 0;

    QuadraticFn(int N) 
        : a(4,           1,           1,
            1, 6 + 0.2 * N,          -1,
            1,          -1, 8 + 0.2 * N),
          b(1, -2, 3),
          c(N)
    {
    }

    Vec3 grad(const Vec3 &x) const {
        return a * x + b;
    }

    double operator()(const Vec3 &x) const {
        return 0.5 * x * (a * x) + b * x + c;
    }
};

const double EPS = 1e-6;

void mngs(const QuadraticFn &f) {
    Vec3 x;
    double dm = f.a.diag_dom_magnitude();

    for (int iter = 0; iter < 1000; ++iter) {
        printf("%02d (%8.4f, %8.4f, %8.4f) f(x) = %8.4f\n", 
                iter, x.x, x.y, x.z, f(x));

        Vec3 q = f.grad(x);
        if (q.norm() / dm < EPS)
            break;

        double nu = -q.norm_squared() / (q * (f.a * q));
        x = x + nu * q;
    }
}

void coord(const QuadraticFn &f) {
    double dm = f.a.diag_dom_magnitude();

    const Vec3 orts[3] = {
        Vec3(1, 0, 0),
        Vec3(0, 1, 0),
        Vec3(0, 0, 1)
    };

    Vec3 x;

    for (int iter = 0; iter < 10; ++iter) {
        printf("%02d (%8.4f, %8.4f, %8.4f) f(x) = %8.4f\n", 
                iter, x.x, x.y, x.z, f(x));

        if ((f.a * x + f.b).norm() / dm < EPS)
            break;

        Vec3 q = orts[iter % 3];
        double nu = -q * (f.a * x + f.b);
        nu /= q * (f.a * q);

        x = x + nu * q;
    }
}

int main() {
    QuadraticFn f(0);
    double dm = f.a.diag_dom_magnitude();
    printf("%f\n", dm);

    printf("gradient\n");
    mngs(f);
    printf("coord\n");
    coord(f);
}

