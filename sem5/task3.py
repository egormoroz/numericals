import numpy as np
import matplotlib.pyplot as plt


def lsm_normal(x, y, n):
    Q = np.vander(x, N=n+1)
    return np.linalg.solve(Q.T @ Q, Q.T @ y)


def next_orth_poly(x, qcur, qprev):
    qcur_x = np.polyval(qcur, x)
    alpha = x @ qcur_x**2 / (qcur_x@qcur_x)

    qprev_x = np.polyval(qprev, x)
    beta = np.sum(x * qcur_x * qprev_x) / (qprev_x@qprev_x)

    qnext = np.hstack((qcur, 0))
    qnext = np.polyadd(qnext, -alpha*qcur)
    qnext = np.polyadd(qnext, -beta*qprev)

    return qnext


def poly_gen(x):
    qprev = np.array([1])
    qcur = np.array([1, -np.sum(x) / len(x)])

    yield qprev
    yield qcur

    while True:
        qcur, qprev = next_orth_poly(x, qcur, qprev), qcur
        yield qcur


def lsm_orth(x, y, n):
    n += 1
    E = np.zeros((len(x), n))
    q = [p for _, p in zip(range(n), poly_gen(x))]
    for i, p in enumerate(q):
        E[:, i] = np.polyval(p, x)

    y = E.T @ y
    poly = np.zeros(n)
    for i, p in enumerate(q):
        a_i = y[i] / (E[:, i] @ E[:, i])
        poly = np.polyadd(poly, a_i * p)

    return poly


def f(x):
    return x * np.sin(x)


def gen_table(n, a, b, k=3, err_scale=1e-2):
    x_ideal = np.linspace(a, b, n)
    y_ideal = f(x_ideal)
    x = np.repeat(x_ideal, k)
    y = np.repeat(y_ideal, k) + (1 - 2 * np.random.random(len(x))) * err_scale
    return x_ideal, y_ideal, x, y


def square_error(p, x, y_ideal):
    error = np.polyval(p, x) - y_ideal
    return error @ error


m, max_n = 50, 25
a, b = 0, 1
x_ideal, y_ideal, x, y = gen_table(m, a, b, err_scale=0.1)

x_lp = np.linspace(a, b)

for n in range(1, 5+1):
    a_normal = lsm_normal(x, y, n)
    a_orth = lsm_orth(x, y, n)

    plt.plot(x_lp, np.polyval(a_normal, x_lp),
             x_lp, np.polyval(a_orth, x_lp),
             x, y, '.')
             # x_ideal, y_ideal, '.')
    plt.title(f'deg={n}')
    plt.show()

print('{:>4} {:>10} {:>10} {:>12} {:>8}'.format(
    'n', 'normal', 'orthog', 'diff', 'improv%'))
for n in range(1, max_n+1):
    a_normal = lsm_normal(x, y, n)
    a_orth = lsm_orth(x, y, n)

    normal_error = square_error(a_normal, x_ideal, y_ideal)
    orth_error = square_error(a_orth, x_ideal, y_ideal)
    diff = normal_error - orth_error
    improv = np.round(100 * diff / normal_error, decimals=2)

    print('{: 4d} {:10.4e} {:10.4e} {:12.4e} {:8.2f}'.format(
        n, normal_error, orth_error, diff, improv))
