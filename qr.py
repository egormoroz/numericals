import numpy as np

def qr_decomp(A):
    n_rows, n_cols = A.shape
    assert n_rows == n_cols

    A_orig = A
    P = np.eye(n_rows)
    for i in range(n_rows - 1):
        a = A[0]
        b = np.zeros_like(a)
        b[0] = 1

        u = a + np.sign(a[0]) * np.linalg.norm(a) * b
        n = u / np.linalg.norm(u)

        P_i = np.eye(A.shape[0]) - 2 * n[:, np.newaxis] @ n[np.newaxis, :]

        P_iext = np.eye(n_rows)
        P_iext[i:, i:] = P_i
        P = P_iext @ P
        A = A[1:, 1:]

    return P.T, P @ A_orig


def comp_eigvals(A, eps=1e-8):
    ics = np.tril_indices_from(A, -1)
    for n in range(1000):
        Q, R = qr_decomp(A)
        A = R @ Q

        err = np.max(np.abs(A[ics]))
        if err < eps:
            break

    i = np.diag_indices_from(A)
    return A[i], err, n


A = np.array([
    [1, 2, 3], 
    [2, 1, 4],
    [3, 4, 1]])
print(np.linalg.eigvals(A))

B = comp_eigvals(A)
print(B)

