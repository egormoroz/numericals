import numpy as np
from math import pi

def ipoly_lagr(x_table, y_table, x):
    y, n = 0, len(x_table)
    for i in range(n):
        num, denum = 1, 1
        for j in range(n):
            if i != j:
                num *= x - x_table[j]
                denum *= x_table[i] - x_table[j]
        y += num / denum * y_table[i]

    return y


def ipoly_newt(x_table, y_table, x):
    y, n = 0, len(x_table)
    for i in range(n):
        omega = 1
        for j in range(i):
            omega *= x - x_table[j]
        
        dd = 0
        for j in range(i+1):
            denum = 1
            for k in range(i+1):
                if j != k:
                    denum *= x_table[j] - x_table[k]
            dd += y_table[j] / denum

        y += dd * omega

    return y


def alterspace(a, b, n=100):
    x = np.cos((2*np.arange(n, dtype=np.float64) + 1) / (2*n) * pi)
    x *= b - a
    x += b + a
    x *= 0.5
    return np.flip(x)


def run1(a, b, f, ipoly, N=50, M=1000):
    x_test = np.linspace(a, b, M)
    y_test = f(x_test)
    max_diff = lambda y: np.max(np.abs(y - y_test))

    for n in range(1, N+1):
        x_table = np.linspace(a, b, n)
        y_table = f(x_table)
        y_ip = ipoly(x_table, y_table, x_test)

        x_table = alterspace(a, b, n)
        y_table = f(x_table)
        y_ipopt = ipoly(x_table, y_table, x_test)

        print('{:02d} {} {:.4e} {:.4e}'.format(
            n, M, max_diff(y_ip), max_diff(y_ipopt)))


def main():
    a, b = 1, 2
    f = lambda x: x + np.log10(x/5)

    print("---------lagrange-------\n");
    run1(a, b, f, ipoly_lagr)

    print("----------newton--------\n");
    run1(a, b, f, ipoly_newt)


if __name__ == '__main__':
    main()

