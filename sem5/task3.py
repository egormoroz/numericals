import numpy as np
import matplotlib.pyplot as plt


'''
найти коэф-ы аппроксимирующего полинома 
с помощью метода наим. квадратов
Аргументы:
    x, y - табличные точки и значения в этих точках
    n - степень аппроксимирующего полинома
'''
def lsm_normal(x, y, n):
    Q = np.vander(x, N=n+1)
    return np.linalg.solve(Q.T @ Q, Q.T @ y)


'''
Найти следующий ортагональный полином согласно формуле
q_{k+1} = x * q_k - alpha * q_k - beta * q_{k-1},
Аргументы:
    x - табличные точки {x_i | i = 0, n-1}
    qcur - коэффициенты полинома q_k, от старшего к младшему
    qprev - коэффициенты полинома q_{k-1}, от старшего к младшему
Возвращает:
    коэф-ы q_{k+1}
'''
def next_orth_poly(x, qcur, qprev):
    # значения полинома q_k в табличных точках x, т.е. qcur_x = q_k(x)
    qcur_x = np.polyval(qcur, x)
    # значения полинома q_{k-1} в табличных точках x, т.е. qprev_x = q_{k-1}(x)
    qprev_x = np.polyval(qprev, x)

    # <u, v> - скалярное произвденеие u и v

    # коэф. альфа = <x, q_k(x)^2> / <q_k(x), q_k(x)>
    alpha = x @ qcur_x**2 / (qcur_x@qcur_x)
    # коэф. бета = Сумма(x_i * q_k(x_i) * q_{k-1}(x_i)) / <q_{k-1}(x), q_{k-1}(x)>
    beta = np.sum(x * qcur_x * qprev_x) / (qprev_x@qprev_x)

    # следующий полином qnext = q_{k+1} по формуле:
    # qnext = x * qcur - alpha * qcur - beta * qprev

    # hstack объединияет несколько массивов в один
    # здесь мы добавляем в конец массива коэффициентов qcur 0,
    # умножая тем самым qcur на x. Например:
    # q_k(x) = x + 2, qcur = [1, 2]
    # x * q_k(x) = x^2 + 2x = [1, 2, 0]
    qnext = np.hstack((qcur, 0))

    # с помощью нумпая к qnext добавляем -alpha * q_k(x)
    qnext = np.polyadd(qnext, -alpha*qcur)
    # аналогично
    qnext = np.polyadd(qnext, -beta*qprev)

    return qnext


'''
Возвращает генератор ортогональных полиномов, например
for i, q_i in zip(range(2), poly_gen([1, 3])):
    print(q_i)
Выведет коэфы первых двух полиномов:
    [1] (q_0 = 1)
    [1, -2] (q_1 = x - 2)
'''
def poly_gen(x):
    qprev = np.array([1])
    qcur = np.array([1, -np.sum(x) / len(x)])

    yield qprev
    yield qcur

    while True:
        qcur, qprev = next_orth_poly(x, qcur, qprev), qcur
        yield qcur


# Построить аппроксимирующий полином степени n
# с помощью ортогональных полиномов 
def lsm_orth(x, y, n):
    n += 1

    # матрица E из методички, по смыслу похожа на матрицу Вандермонда
    E = np.zeros((len(x), n))
    # положить в q первые n ортогональных полиномов
    q = [p for _, p in zip(range(n), poly_gen(x))]
    for i, p in enumerate(q):
        # в i-том столбце находится q_i(x), здесь p = q_i
        E[:, i] = np.polyval(p, x)

    y = E.T @ y
    poly = np.zeros(n)
    # матрица СЛАУ диагональная, так что решаём СЛАУ руками
    for i, p in enumerate(q):
        a_i = y[i] / (E[:, i] @ E[:, i])
        # аппрокс. полином = Сумма(a_i * q_i), здесь p = q_i
        poly = np.polyadd(poly, a_i * p)

    return poly


def f(x):
    return x * np.sin(x)


'''
Сгенерировать эксперементальные данные; для этого
1. x_ideal <- n равноудалённых точек на [a, b]
2. y_ideal <- f(x_ideal)
3. x <- повторить каждую точку x_ideal_i k раз, len(x) = n * k
4. y <- повторить каждый y_ideal_i k раз
5. y <- y + (случ_величины из [-1; 1]) * err_scale, err_scale - масштаб погрешности

Пример:
    n = 3, a = 1, b = 3, k = 3, err_scale = 1, f(x) = x^2
    1. x_ideal = [1, 2, 3]
    2. y_ideal = [1, 4, 9]
    3. x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    4. y = [1, 1, 1, 4, 4, 4, 9, 9, 9]
    5. y = y + [-0.8, 0.1, 1, -1, -0.5, ...];
       y = [0.2, 1.1, 2, 3, 3.5, ...]
'''
def gen_table(n, a, b, k=3, err_scale=1e-2):
    x_ideal = np.linspace(a, b, n)
    y_ideal = f(x_ideal)
    x = np.repeat(x_ideal, k)
    y = np.repeat(y_ideal, k) + (1 - 2 * np.random.random(len(x))) * err_scale
    return x_ideal, y_ideal, x, y


# посчитать ошибку по методу наименьших квадратов
def square_error(p, x, y_ideal):
    error = np.polyval(p, x) - y_ideal
    return error @ error


m, max_n = 50, 30
a, b = -1, 1
x_ideal, y_ideal, x, y = gen_table(m, a, b, err_scale=0.1)

x_lp = np.linspace(a, b)

# построить графики для первых 5 степеней аппрокс. полинома
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
