import numpy as np


def stabilize(P, Q, ideal_eigvals):
    n, _ = P.shape
    S = np.hstack([np.linalg.matrix_power(P, i) @ Q for i in range(n)])
    assert np.linalg.matrix_rank(S) == n

    PP = np.linalg.inv(S) @ P @ S

    poly = np.poly(np.linalg.eigvals(PP))
    ideal_poly = np.poly(ideal_eigvals)

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i:] = poly[:-i-1]

    gamma = poly[1:] - ideal_poly[1:]

    return gamma[np.newaxis, :] @ np.linalg.inv(S @ K)


P = np.array([[4, 0], [1, 1]])
Q = np.array([[1, 0]]).T
ideal_eigvals = np.array([-1, -1])

C = stabilize(P, Q, ideal_eigvals)
print('--- C ---')
print(C)
print('--- eig_vals(PQ + C) ---')
print(np.linalg.eigvals(P + Q @ C))
