import numpy as np
import matplotlib.pyplot as plt
import scipy


def exact_sol(x):
    y1 = A**1.5*B**-.5*np.pi*np.sin((A*B)**.5*x) + B*np.pi*np.cos((A*B)**.5*x)
    y2 = A*np.pi*np.cos((A*B)**.5*x) - A**-.5*B**1.5*np.pi*np.sin((A*B)**.5*x)

    return np.vstack([y1, y2]).T


def get_scheme_params(c2):
    return c2, 1/(2*c2), 1 - 1/(2*c2), c2


def f(x, y):
    global f_counter
    f_counter += 1
    return np.array([A * y[1], -B * y[0]])


def estm_initial_h(s):
    delta = (1 / np.max(np.abs([x0, x_target]))) ** (s+1)
    delta += np.linalg.norm(f(x0, y0))**(s+1)
    initial_h = (eps / delta) ** (1/(s+1))
    return initial_h


def rk2_step(xk, yk, h, f_xy):
    return yk + b1*h*f_xy + b2*h*f(xk + c2*h, yk + a21*h*f_xy)


def rk4_step(xk, yk, h, f_xy):
    k1 = h*f_xy
    k2 = h*f(xk + h/2, yk + k1/2)
    k3 = h*f(xk + h/2, yk + k2/2)
    k4 = h*f(xk + h, yk + k3)
    return yk + 1/6 * (k1 + 2*k2 + 2*k3 + k4)


def rk_const(rk_step, h):
    xk, yk = x0, y0

    x = [x0]
    y = [y0]
    while xk < x_target:
        h = min(h, x_target - xk)
        yk = rk_step(xk, yk, h, f(xk,yk))
        xk += h
        x.append(xk)
        y.append(yk)

    return np.array(x), np.array(y)


def rk_adaptive(rk_step, initial_h):
    xk, yk = x0, y0
    h = initial_h

    x = [x0]
    y = [y0]
    err_norms = [0]
    while xk < x_target:
        h = min(h, x_target - xk)
        f_xy = f(xk, yk)
        while True:
            y_full = rk_step(xk, yk, h, f_xy)
            y_half = rk_step(xk, yk, h/2, f_xy)
            y_2half = rk_step(xk+h/2, y_half, h/2, f(xk+h/2, y_half))

            err = (y_2half - y_full) / (1 - 2**-s)
            err_norm = np.linalg.norm(err)

            if err_norm > ro * 2**s:
                h /= 2
                continue

            if err_norm > ro and err_norm <= ro * 2**s:
                yk = y_2half
                xk += h
                h /= 2

                # NEW
                err = (y_2half - y_full) / (2**s - 1)
                err_norm = np.linalg.norm(err)
                break

            if ro / 2**(s+1) <= err_norm and err_norm <= ro:
                yk = y_full
                xk += h
                break

            assert err_norm < ro / 2**(s+1)
            yk = y_full
            xk += h
            h *= 2
            break
        x.append(xk)
        y.append(yk)
        err_norms.append(err_norm)

    return np.array(x), np.array(y), np.array(err_norms)


A, B = 1/15, 1/25
ksi = 1/11

x0 = 0
y0 = np.array([B, A]) * np.pi

x_target = np.pi

eps, ro = 1e-4, 1e-5
f_counter = 0

a21, b2, b1, c2 = get_scheme_params(ksi)
print('y(x+h) = y(x) + b1*h*f(x,y(x)) + b2*h*f(x+c2*h, y(x)+a21*h*f(x,y(x)))')
print(f'{b1=:.2f}, {b2=:.2f}, {c2=:.2f}, {a21=:2f}')

exact_y = exact_sol(x_target)
print(f'exact_sol: {exact_y}')
print()

# Посчитаем начальный шаг
initial_h2 = estm_initial_h(2)
initial_h4 = estm_initial_h(4)
print(f'{initial_h2=:.4f}')
print(f'{initial_h4=:.4f}\n')

# -------RK2-------

# Посчитаем два раза, чтобы оценить погрешность с помощью метода Рунге
s = 2
_, y1 = rk_const(rk2_step, initial_h2)
n = f_counter
_, y2 = rk_const(rk2_step, initial_h2/2)
err = (y2[-1] - y1[-1]) / (2**s - 1)

# y2[-1] += err
print(f'rk2_const_step: {y2[-1]}, ||R|| = {np.linalg.norm(err)},',
      f'{n} evaluations ({f_counter})')
print(np.linalg.norm(y1[-1] - exact_y))

f_counter = 0
_, y, _ = rk_adaptive(rk2_step, initial_h2)
print(f'rk2_adaptive_step: {y[-1]}, {f_counter} evaluations')
print(np.linalg.norm(y[-1] - exact_y))
print()

# -------RK4-------

f_counter = 0
_, y = rk_const(rk4_step, initial_h4)
print(f'rk4_const_step: {y[-1]}, {f_counter} evaluations')
print(np.linalg.norm(y[-1] - exact_y))

f_counter = 0
_, y, _ = rk_adaptive(rk4_step, initial_h4)
print(f'rk4_adaptive_step: {y[-1]}, {f_counter} evaluations')
print(np.linalg.norm(y[-1] - exact_y))



h_opt = initial_h2
for h in np.linspace(initial_h2, x_target, 10):
    _, y = rk_const(rk2_step, h)
    err_norm = np.linalg.norm(exact_y - y[-1])
    if err_norm > eps:
        break
    h_opt = h

print(f'\n{h_opt=:.2f}')
x, y2_const = rk_const(rk2_step, h_opt)
x, y4_const = rk_const(rk4_step, h_opt)
y_ideal = exact_sol(x)

rk2_error = np.log10(np.linalg.norm(y2_const[1:] - y_ideal[1:], axis=1))
rk4_error = np.log10(np.linalg.norm(y4_const[1:] - y_ideal[1:], axis=1))
plt.plot(x[1:], rk2_error, label='rk2')
plt.plot(x[1:], rk4_error, label='rk4')
plt.legend()
plt.ylabel('log10(||y - y_ideal||)')
plt.title('abs error')
plt.show()


x, y, err_estms = rk_adaptive(rk2_step, initial_h2)
hs = x[1:] - x[:-1]

abs_errs = np.zeros_like(err_estms)
for i in range(1, len(x)):
    y_scipy = scipy.integrate.solve_ivp(f, [x[i-1], x[i]], y[i-1]).y[:, -1]
    abs_errs[i] = np.linalg.norm(y[i] - y_scipy)

# abs_errs = np.linalg.norm(y - y_ideal, axis=1)

fig, axs = plt.subplots(3, 1)
axs[0].plot(x[:-1], hs)
axs[0].set_xlabel('x_i')
axs[0].set_ylabel('h_i')

axs[1].plot(x[1:], abs_errs[1:]/err_estms[1:])
axs[1].set_xlabel('x_i')
axs[1].set_ylabel('abs_err/err_estm')

f_counts2 = []
f_counts4 = []
ks = np.arange(-8, 0)
for k in ks:
    eps = 10.0**k
    ro = eps / 10
    f_counter = 0
    initial_h = estm_initial_h(2)
    rk_adaptive(rk2_step, initial_h)
    f_counts2.append(f_counter)
    initial_h = estm_initial_h(4)
    f_counter = 0
    rk_adaptive(rk4_step, initial_h)
    f_counts4.append(f_counter)

axs[2].plot(ks, f_counts2, label='rk2')
axs[2].plot(ks, f_counts4, label='rk4')
axs[2].set_xlabel('log10(eps)')
axs[2].set_ylabel('f_count')

fig.tight_layout()
plt.legend()
plt.show()
