import numpy as np
import matplotlib.pyplot as plt
from math import pi

def alterspace(a, b, n=100):
    x = np.cos((2*np.arange(n, dtype=np.float64) + 1) / (2*n) * pi)
    x *= b - a
    x += b + a
    x *= 0.5
    return np.flip(x)

def spline10(xs, ys, x):
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    y = (x - xs[i]) / (xs[i-1] - xs[i]) * ys[i-1]
    y += (x - xs[i-1]) / (xs[i] - xs[i-1]) * ys[i]
    return y

def build_spline21(xs, ys):
    n = len(xs) - 1
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)

    for i in range(n):
        A[3*i + 0, 3*i:3*i + 3] = [1, xs[i], xs[i]**2]
        A[3*i + 1, 3*i:3*i + 3] = [1, xs[i+1], xs[i+1]**2]
        if i + 1 < n:
            A[3*i + 2, 3*i:3*i + 6] = [
                0, 1, 2*xs[i+1], 0, -1, -2*xs[i+1]]

        b[3*i:3*i+2] = [ys[i], ys[i+1]]

    A[-1, -2:] = [1, 2*xs[-1]]
    b[-1] = 0

    p = np.linalg.solve(A, b).reshape((n, 3))
    return np.flip(p, axis=1).T

def build_spline32(xs, ys, dfa, dfb):
    n = len(xs) - 1
    A = np.zeros((n*4, n*4))
    b = np.zeros(n*4)

    for i in range(n):
        b[4*i:4*i+2] = [ys[i], ys[i+1]]

        A[4*i + 0, 4*i:4*i+4] = [1, xs[i], xs[i]**2, xs[i]**3]
        A[4*i + 1, 4*i:4*i+4] = [1, xs[i+1], xs[i+1]**2, xs[i+1]**3]

        if i + 1 == n:
            continue
        A[4*i + 2, 4*i:4*i+8] = [
            0, 1, 2*xs[i+1], 3*xs[i+1]**2, 
            0, -1, -2*xs[i+1], -3*xs[i+1]**2, 
        ]
        A[4*i + 3, 4*i:4*i+8] = [
            0, 0, 2, 6*xs[i+1],
            0, 0, -2, -6*xs[i+1],
        ]

    A[-2, :4] = [0, 1, 2*xs[0], 3*xs[0]**2]
    A[-1, -4:] = [0, 1, 2*xs[-1], 3*xs[-1]**2]
    b[-2] = dfa
    b[-1] = dfb

    p = np.linalg.solve(A, b).reshape((n, 4))
    return np.flip(p, axis=1).T

def spline(xs, p, x):
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    return np.polyval(p[:, i], x)

def f(x):
    return x + np.log10(x/5)

def derf(x):
    return 1 + 1/(np.log(10) * x)

def run(a, b, max_n, k, spl):
    x_test = np.linspace(a, b, k)
    y_test = f(x_test)
    for n in range(2, max_n+1):
        xs = np.linspace(a, b, n)
        iy_test = spl(xs, f(xs), x_test)
        dev = np.max(np.abs(y_test - iy_test))

        xs = alterspace(a, b, n)
        iy_test = spl(xs, f(xs), x_test)
        dev_opt = np.max(np.abs(y_test - iy_test))

        print('{:02d} {} {:.4e} {:.4e}'.format(
                n, k, dev, dev_opt))


a, b = 1, 2
max_n, k = 50, 1000

def spl21(xs, ys, x):
    return spline(xs, build_spline21(xs, ys), x)

def spl32(xs, ys, x):
    p = build_spline32(xs, ys, derf(a), derf(b))
    return spline(xs, p, x)

print('S10')
run(a, b, max_n, k, spline10)
print('S21')
run(a, b, max_n, k, spl21)
print('S32')
run(a, b, max_n, k, spl32)

