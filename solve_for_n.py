import numpy as np
import scipy.optimize as opt
from pulp import *


def solve_for_n(balances: list):
    n = len(balances)

    def constraint1(vec):
        vec = np.reshape(vec, (n, n))
        vecteur_uns = np.ones((n))

        product = np.dot(vec, vecteur_uns)

        return product - np.array(balances)

    def constraint_anti_symmetric(vec):
        matrix = np.reshape(vec, (n, n))
        diff = matrix + matrix.T
        return np.linalg.norm(diff.flatten(), ord=2)

    def minimize_vector(vector1):
        """Minimize sum and number of non-zero elements in vector1."""
        return np.linalg.norm(vector1, ord=1)

    initial_guess = np.ones(n**2)

    result = opt.minimize(minimize_vector, x0=initial_guess, method="SLSQP", constraints=[
        {'type': 'eq', 'fun': constraint1},
        {'type': 'eq', 'fun': constraint_anti_symmetric},
    ], options={'maxiter': 3000, 'disp': True})

    argument_result = np.reshape(result.x, (n, n))
    argument_result = np.round(argument_result, 2)

    return (argument_result, result.fun, result.message, result.success)


def solve_for_n_linalg(balances=list):
    if not np.isclose(np.sum(balances), 0):
        raise ValueError("Entries of b must sum to 0")

    n = len(balances)
    k = n * (n - 1)

    # Solve linear programming problem
    c = np.ones(k)
    A_eq = get_const_matrix(n)

    result = opt.linprog(c, A_eq=A_eq, b_eq=np.asarray(balances),
                         method='simplex',
                         #  bounds=bounds,
                         options={
                             'disp': False,
                             "tol": 1e-2
    })

    return (np.round(vec_2_mat(result.x).T, 2), result.fun, result.message, result.success)


def solve_for_n_pulp(balances):
    if not np.isclose(np.sum(balances), 0):
        raise ValueError("Entries of balances must sum to 0")

    n = len(balances)
    k = n * (n - 1)

    # Create the problem
    prob = LpProblem("LP_Problem", LpMinimize)

    # Decision variables
    x = LpVariable.dicts("x", range(k), lowBound=0)

    # Objective function
    prob += lpSum([x[i] for i in range(k)])

    # Constraints
    A_eq = get_const_matrix(n)
    for i in range(n):
        prob += (lpSum([A_eq[i, j] * x[j]
                 for j in range(k)]) == balances[i])

    # Solve the problem
    prob.solve(HiGHS())

    # Extract the results
    result_x = np.array([value(x[i]) for i in range(k)])
    return (np.round(vec_2_mat(result_x).T, 2), value(prob.objective), LpStatus[prob.status], prob.status == LpStatusOptimal)

# def get_const_matrix(n):
#     if n < 2:
#         raise ValueError("n must be at least 2")

#     block = np.vstack(
#         (np.ones((1, n-1)), -np.eye(n-1)))
#     b_list = [None] * n
#     b_list[0] = block

#     for i in range(1, n):
#         block[[i-1, i], :] = block[[i, i-1], :]
#         b_list[i] = block.copy()

#     return np.hstack(b_list)


def get_const_matrix(n):
    assert n >= 2, "Number of counterparties must be at least 2"

    # Create the initial block matrix
    block = np.vstack((np.ones((1, n-1)), -np.eye(n-1)))

    # Initialize the list to hold the blocks
    b_list = [None] * n
    b_list[0] = block

    # Fill the list with the appropriate blocks
    for i in range(1, n):
        block[[i-1, i], :] = block[[i, i-1], :]
        b_list[i] = block.copy()

    # Concatenate the blocks horizontally to form the final matrix
    return np.hstack(b_list)


def mat_2_vec(X):
    # Get the lower and upper triangular parts of the matrix, excluding the diagonal
    lower_tri = np.tril(X, -1)
    upper_tri = np.triu(X, 1)

    # Combine the lower and upper triangular parts and flatten to a vector
    return np.concatenate((lower_tri[lower_tri != 0], upper_tri[upper_tri != 0]))


def vec_2_mat(x):
    # Calculate the size of the matrix
    n = int(0.5 * (1 + np.sqrt(1 + 4 * len(x))))

    # Create an n x n matrix filled with zeros
    X = np.zeros((n, n))

    # Fill the lower and upper triangular parts of the matrix with the vector elements
    X[np.tril_indices(n, -1)] = x[:len(x)//2]
    X[np.triu_indices(n, 1)] = x[len(x)//2:]

    return X
