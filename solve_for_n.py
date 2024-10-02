import numpy as np
import scipy.optimize as opt


def solve_for_n(balances=list):
    print('yolo')
    if not np.isclose(np.sum(balances), 0):
        raise ValueError("Entries of b must sum to 0")

    n = len(balances)
    k = n * (n - 1)

    # Solve linear programming problem
    c = np.ones(k)
    A_eq = get_const_matrix(n)
    bounds = [(0, None)] * k

    result = opt.linprog(c, A_eq=A_eq, b_eq=balances,
                         bounds=bounds)

    if not result.success:
        raise ValueError("Linear programming did not converge")

    return (np.round(vec_2_mat(result.x)), result.fun, result.message, result.success)


def get_const_matrix(n):
    if n < 2:
        raise ValueError("n must be at least 2")

    block = np.vstack((np.ones((1, n-1)), -np.eye(n-1)))
    b_list = [None] * n
    b_list[0] = block

    for i in range(1, n):
        block[[i-1, i], :] = block[[i, i-1], :]
        b_list[i] = block.copy()

    return np.hstack(b_list)


def mat_2_vec(X):
    return X[np.tril_indices_from(X, k=-1)] + X[np.triu_indices_from(X, k=1)]


def vec_2_mat(x):
    n = int(0.5 * (1 + np.sqrt(1 + 4 * len(x))))
    X = np.zeros((n, n))
    lower_indices = np.tril_indices(n, -1)
    upper_indices = np.triu_indices(n, 1)
    X[lower_indices] = x[:len(lower_indices[0])]
    X[upper_indices] = x[len(lower_indices[0]):]
    return X
