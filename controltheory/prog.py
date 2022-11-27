import numpy as np
import math
from math import exp

G = lambda t: np.array([[exp(-2*t), exp(-3*t)], [exp(-3*t), exp(-4*t)]])

def matexp_tailor(a, t, eps, max_terms):
    n = a.shape[0]

    b = np.eye(n)
    s = np.eye(n)
    k = 1

    for i in range(1, max_terms):
        b = b @ a
        k *= t / i
        term = b * k
        s += term

        if np.linalg.norm(term) < eps:
            break

    return s


def matexp(a, t, eps=1e-12, max_terms=100):
    if abs(t) < eps:
        return np.eye(a.shape[0])

    p = max(0, math.ceil(np.log2(abs(t) * np.linalg.norm(a))))
    b = matexp_tailor(a, t / 2**p, eps, max_terms)
    for _ in range(p):
        b = b @ b
    return b


def int_simpson(f, a, b, n=16):
    assert n % 2 == 0

    h = (b - a) / n
    f_a = f(a)
    s1 = np.zeros_like(f_a)
    s2 = np.zeros_like(f_a)

    for i in range(1, n//2):
        s1 += f(a + 2 * i * h)

    for i in range(1, n//2 + 1):
        s2 += f(a + (2 * i - 1) * h)

    return h/3 * (f_a + 2 * s1 + 4 * s2 + f(b))


def compute_a(P, Q, T):
    def BBT(t):
        B = matexp(P, -t) @ Q(t)
        result = B @ B.T

        return result

    return int_simpson(BBT, 0, T)

def compute_eta(P, T, f, x0, x1):
    eta = matexp(P, -T) @ x1 - x0
    eta -= int_simpson(lambda t: matexp(P, -t) @ f(t), 0, T)

    return eta

P = np.array([[1, 0], [0, 2]])
Q = lambda t: np.array([[1, 1]]).T
f = lambda t: np.array([[0, 0]]).T
x0 = np.array([[0, 0]]).T
x1 = np.array([[1, 1]]).T
T = 1

A = compute_a(P, Q, T)
eta = compute_eta(P, T, f, x0, x1)

print('A', A, sep='\n')
print('eta', eta, sep='\n')

try:
    c = np.linalg.solve(A, eta)
    print('c', c, sep='\n')

    def u(t):
        B = matexp(P, -t) @ Q(t)
        return B.T @ c

    def int_fn(t):
        return matexp(P, -t) @ (Q(t) @ u(t) + f(t))

    x_T = matexp(P, T) @ (x0 + int_simpson(int_fn, 0, T))
    print('x(T)', x_T, sep='\n')

except np.linalg.LinAlgError:
    rank_a = np.linalg.matrix_rank(A, 1e-12)
    a_eta = np.hstack((A, eta))
    rank_aeta = np.linalg.matrix_rank(a_eta, 1e-12)

    print('matrix A is singular and the (x0, x1) is ', end='')
    if rank_a == rank_aeta:
        print('controllable')
    else:
        print('not controllable')

