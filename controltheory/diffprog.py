import numpy as np


def comp_x(P, Q, u, f, x0):
    m, _, _ = Q.shape
    x = x0
    for k in range(m):
        x = P[k] @ x + Q[k] @ u[k] + f[k]
    return x


def create_eqs(P, Q, f, x0, x1):
    m, n, r = Q.shape

    A = Q[0]
    eta = P[0] @ x0 + f[0]

    for k in range(1, m):
        A = np.hstack((P[k] @ A, Q[k]))
        eta = P[k] @ eta + f[k]

    return A, x1 - eta


P = np.array(
    [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0]])
Q = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1]]).T

P = np.repeat(P[np.newaxis, :, :], 2, axis=0)
Q = np.repeat(Q[np.newaxis, :, :], 2, axis=0)


f = np.zeros((2, 4, 1))

x0 = np.zeros((4, 1))
x1 = np.ones((4, 1))

m, n, r = Q.shape

A, eta = create_eqs(P, Q, f, x0, x1)

print('--- A ---')
print(A)
print('--- eta ---')
print(eta)

u, res, rank, _ = np.linalg.lstsq(A, eta, rcond=1e-8)
u = u.reshape((m, r, 1))
print(f'rank(A) = {rank}, ||eta - A@U|| = {np.linalg.norm(res)}')
print('--- u ---')
print(u)

x = comp_x(P, Q, u, f, x0)

print('--- x(m) ---')
print(x)
