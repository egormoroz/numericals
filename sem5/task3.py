import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

def least_square_method(x, y, n):
    Q = np.vander(x, N=n+1)
    return np.linalg.solve(Q.T @ Q, Q.T @ y)

    
def int_simpson(f, a, b, n=16):
    assert n % 2 == 0

    h = (b - a) / n
    f_a = f(a)
    s1 = np.zeros_like(f_a, dtype=np.float64)
    s2 = np.zeros_like(f_a, dtype=np.float64)

    for i in range(1, n//2):
        s1 += f(a + 2 * i * h)

    for i in range(1, n//2 + 1):
        s2 += f(a + (2 * i - 1) * h)

    return h/3 * (f_a + 2 * s1 + 4 * s2 + f(b))


def legendre_coeffs(f, L):
    n = L.shape[0]
    c = np.zeros(n, dtype=np.float64)
    for i in range(n):
        alpha = lambda x: f(x) * np.polyval(L[i], x)
        beta = lambda x: np.polyval(L[i], x)**2
        num = int_simpson(alpha, -1, 1)
        denum = int_simpson(beta, -1, 1)
        c[i] = num/denum

    return c


f = lambda x: x*(x+2)**0.5

L = np.array(
    [[   0,   0,   0,    1],
     [   0,   0,  -1,    0],
     [   0, 1.5,   0, -0.5],
     [-2.5,   0, 1.5,    0]])

x = np.linspace(-1, 1, 5)
y = f(x)
a = least_square_method(x, y, 3)

b = np.zeros(4, dtype=np.float64)
for ci, p in zip(legendre_coeffs(f, L), L):
    b += ci * p

#почему у np.polyval и Polynomial разный порядок коэффов?????????????
lsm_p = str(Polynomial(np.flip(np.round(a, 4))))
lg_p = str(Polynomial(np.flip(np.round(b, 4))))

print(lsm_p.replace('**', '^'))
print(lg_p.replace('**', '^'))

fig, axs = plt.subplots(2)

x = np.linspace(-1, 1, 100)
y = f(x)
u = np.polyval(a, x)
v = np.polyval(b, x)

axs[0].plot(x, y, 'r', label='f(x)')
axs[0].plot(x, u, 'g', label='lsm')
axs[0].plot(x, v, 'b', label='legendre')
axs[0].legend(loc='upper right')

axs[1].plot(x, np.zeros_like(x), '--')
axs[1].plot(x, y-u, 'g', label='f(x) - lsm(x)')
axs[1].plot(x, y-v, 'b', label='f(x) - legendre(x)')
axs[1].legend(loc='upper right')
plt.show()

