import numpy as np


def matrix_polyval(p, X):
    Y = np.zeros(X.shape)
    E = np.eye(X.shape[0])
    for i in p:
        Y = Y @ X + E * i

    return Y


def is_zero(v):
    return v @ v < 1e-8


# процесс ортагонализации по Г-Ш
def gs(A):
    n, m = A.shape
    b = np.zeros((min(n, m), m))
    ics = np.arange(n)

    i, k = 0, 0
    while i < n and is_zero(A[i]):
        i += 1
    if i == n:
        return np.array([[]])

    b[0], ics[0] = A[i], i
    k += 1
    while i < n and k < b.shape[0]:
        coeffs = b[:k] @ A[i] / np.sum(b[:k]*b[:k], 1)
        b[k] = A[i] - coeffs @ b[:k]
        if not is_zero(b[k]):
            ics[k] = i
            k += 1
        i += 1

    return b[:k], ics[:k]


# стабилизация для слуачая полной управляемости
def stabilize_fc(P, S, ideal_eigvals):
    n = P.shape[0]
    P = np.linalg.inv(S) @ P @ S
    poly = np.hstack((1, -P[::-1, -1]))
    ideal_poly = np.poly(ideal_eigvals)

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i:] = poly[:-i-1]

    gamma = poly[1:] - ideal_poly[1:]

    return gamma[np.newaxis, :] @ np.linalg.inv(S @ K)


# стабилизация для неполной управляемости
def stabilize_nfc(P, T, k, ideal_eigvals):
    n = P.shape[0]
    P = np.linalg.inv(T) @ P @ T
    if np.any(np.real(np.linalg.eigvals(P[k:, k:])) >= 0):
        print('system is not stabilizable')
        return None

    P = P[:k, :k]
    poly = np.hstack((1, -P[::-1, -1]))
    ideal_poly = np.poly(ideal_eigvals[:k])
    
    K = np.eye(n)
    for i in range(k):
        K[i, i:k] = poly[:-i-1]

    gamma = np.zeros(n)
    gamma[:k] = poly[1:] - ideal_poly[1:]

    return gamma[np.newaxis, :] @ np.linalg.inv(T @ K)


# стабилизировать для скалярного случая (r=1)
def stabilize(P, Q, ideal_eigvals):
    n, _ = P.shape
    S = np.hstack([np.linalg.matrix_power(P, i) @ Q for i in range(n)])
    SS = np.hstack((S, np.eye(n)))

    _, ics = gs(SS.T)
    k = np.count_nonzero(ics < n)

    if k == n:
        return stabilize_fc(P, S, ideal_eigvals)
    else:
        return stabilize_nfc(P, SS[:, ics], k, ideal_eigvals)


P = np.array([[4, 0], [1, 1]])
Q = np.array([[1, 0]]).T
ideal_eigvals = np.array([-1, -1])

print('--- fc case ---')

C = stabilize(P, Q, ideal_eigvals)
print('--- C ---')
print(C)
print('--- eig_vals(P + QC) ---')
print(np.linalg.eigvals(P + Q @ C))
print(matrix_polyval(np.poly(ideal_eigvals), P + Q @ C))

P = np.array(
   [[-1, 0, 0, 0], 
    [0, -1, 1, -1], 
    [1, 1, 0, 0], 
    [-1, 0, 0, 1]])
Q = np.array([[0, 0, 1, 1]]).T
ideal_eigvals = np.array([-1, -1, -1, -1])

print()
print('--- nfc case ---')

C = stabilize(P, Q, ideal_eigvals)
if C is not None:
    print('--- C ---')
    print(C)
    print('--- eig_vals(P + QC) ---')
    print(np.linalg.eigvals(P + Q @ C))
    print(matrix_polyval(np.poly(ideal_eigvals), P + Q @ C))

