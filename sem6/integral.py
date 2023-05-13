import numpy as np
from scipy.integrate import quad
from math import factorial, acos, pi, cos, log, ceil

import matplotlib.pyplot as plt


def left_rect(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + i * h) * h
    return s


def avg_rect(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + i * h + 0.5 * h) * h
    return s


def trapeze(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        a_i, b_i = a + i * h, a + (i+1) * h
        s += (f(a_i) + f(b_i)) * h / 2
    return s


def simpson(f, a, b, n):
    h = (b - a) / n
    if n % 2 != 0:
        s = h / 6 * (f(b - h) + 4 * f(b - 0.5*h) + f(b))
        return s + simpson(f, a, b - h, n - 1) if n > 1 else s

    s1, s2 = 0, 0
    for i in range(1, n // 2 + 1):
        s1 += f(a + (2 * i - 1) * h)
    for i in range(1, n // 2):
        s2 += f(a + 2 * i * h)

    return h / 3 * (f(a) + 4 * s1 + 2 * s2 + f(b))


# для случая beta = 0 посчитали m-ый момент аналитически
def get_mu(a_i, b_i, m):
    global a, alpha

    s = 0
    for k in range(m+1):
        c_mk = factorial(m) / (factorial(k) * factorial(m-k))
        d = (m - k - alpha + 1)
        s += c_mk * a**k * ((b_i - a)**d - (a_i - a)**d) / d
    return s


def solve_poly(a, b, c, d):
    q = 0.5 * (2*(b/(3*a))**3 - b*c/(3*a**2) + d / a)
    p = 1/3 * (3*a*c - b**2) / (3*a**2)
    assert q**2 + p**3 < 0

    r = abs(p)**0.5 * (1 if q >= 0 else -1)
    phi = acos(q/r**3)

    y = 2 * r * np.array([-cos(phi/3), cos(pi/3 - phi/3), cos(pi/3 + phi/3)])
    return y - b/(3*a)


def newt_kot_inner(a_i, b_i):
    M = np.zeros((3, 3))
    mu = np.array([get_mu(a_i, b_i, i) for i in range(3)])

    x = np.array([a_i, (a_i+b_i)/2, b_i])
    for i in range(3):
        M[i] = x ** i
    A = np.linalg.solve(M, mu)
    return A @ f(x)


def gauss_inner(a_i, b_i):
    mu = np.array([get_mu(a_i, b_i, i) for i in range(6)])
    T = np.zeros((3, 3))
    for s in range(3):
        for j in range(3):
            T[s, j] = mu[j + s]

    a = np.linalg.solve(T, -mu[3:])
    x = solve_poly(1, a[2], a[1], a[0])
    
    T = np.zeros((3, 3))
    for i in range(3):
        T[i] = x**i
    A = np.linalg.solve(T, mu[:3])

    return A @ f(x)


def iwrapper(inner, a, b, n):
    h = (b-a)/n
    s = 0
    for i in range(n):
        s += inner(a + i*h, a + (i+1)*h)
    return s


def estm_richardson(S, h, m):
    n = len(S)
    A = np.zeros((n, n))
    S = np.array(S)
    h = np.array(h)

    A[:, -1] = -np.ones(n)
    for i in range(n-1):
        A[:, i] = h**(m+i)

    #CJ[0..n-1] - коэфы C_k и CJ[n] - уточнение по Ричардсону
    CJ = np.linalg.solve(A, -S)

    # Rh и J
    return abs(CJ[-1] - S[-1]), CJ[-1]


def estm_eitk(S, L):
    return -log(abs((S[2] - S[1]) / (S[1] - S[0]))) / log(L)


a, b, alpha, beta = 2.1, 3.3, 2/5, 0
eps = 1e-6


def f(x):
    y = 4.5 * np.cos(7*x) * np.exp(-2/3*x)
    y += 1.4 * np.sin(1.5 * x) * np.exp(-1/3 * x)
    y += 3
    return y


J_ideal = quad(f, a, b)[0]
print('scipy.integrate.quad', J_ideal)
print()

print('left_rect', left_rect(f, a, b, 16))
print('avg_rect', avg_rect(f, a, b, 16))
print('trapeze', trapeze(f, a, b, 16))
print('simpson', simpson(f, a, b, 16))
print()


_, axs = plt.subplots(2, 2)
k = np.arange(2, 512)

names = ['left_rect', 'avg_rect', 'trapeze', 'simpson']
methods = [left_rect, avg_rect, trapeze, simpson]
for i, (method_name, method) in enumerate(zip(names, methods)):
    errs = np.array([abs(method(f, a, b, n) - J_ideal) for n in k])
    ax = axs[i // 2, i % 2]
    ax.plot(k, np.log10(errs))
    ax.set_title(method_name)
    ax.set_xlabel('n')
    ax.set_ylabel('log10(abs_err)')

plt.show()

k = np.arange(2, 128)
J_ideal = iwrapper(gauss_inner, a, b, 512)
names = ['newt', 'gauss']
inners = [newt_kot_inner, gauss_inner]
_, axs = plt.subplots(1, 2)
for i, (inner_name, inner) in enumerate(zip(names, inners)):
    errs = np.array([abs(iwrapper(inner, a, b, n) - J_ideal) for n in k])
    axs[i].plot(k, np.log10(errs))
    axs[i].set_title(inner_name)
    axs[i].set_xlabel('n')
    axs[i].set_ylabel('log10(abs_err)')

plt.show()
    
for i, (inner_name, inner) in enumerate(zip(names, inners)):
    h = [(b-a) / 2**i for i in [0, 1, 2]]
    S = [iwrapper(inner, a, b, 2**i) for i in [0, 1, 2]]
    k = 2
    while True:
        m = estm_eitk(S[-3:], 2)
        err, J = estm_richardson(S, h, m)

        if err < eps:
            break

        k += 1
        h.append((b-a) / 2**k)
        S.append(iwrapper(inner, a, b, 2**k))

    print(f'1. {inner_name} h={h[-1]:.4f} err={err:.4e} m={m:.2f} J={J:.8f}')

print()
for i, (inner_name, inner) in enumerate(zip(names, inners)):
    S = [iwrapper(inner, a, b, 2**i) for i in [0, 1, 2]]
    h_prev, h_opt = (b-a) / 2, (b-a) / 4
    m = estm_eitk(S, 2)
    while True:
        L = h_prev / h_opt
        Rh2 = abs(S[-1] - S[-2]) / (L**m - 1)

        if Rh2 < eps:
            break

        h_prev = h_opt
        h_opt = h_opt * (eps / Rh2) ** (1/m)
        S.append(iwrapper(inner, a, b, int(ceil((b-a) / h_opt))))

    print(f'2. {inner_name} h={h_opt:.4f} err={Rh2:.4e} m={m:.2f} J={S[-1]:.8f}')

