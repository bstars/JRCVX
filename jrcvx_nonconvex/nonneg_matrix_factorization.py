"""
This file solves the positive matrix factorization problem

    min(X,Y).   ||A - XY||_F^2
    s.t.        X_i,j >= 0
                Y_i,j >= 0

with alternating convex optimization
"""

import sys
sys.path.append('..')

import numpy as np


def nonnegative_quadratic_programming_log_barrier(t, P, q, r, xstart=None, ALPHA=0.01, BETA=0.5, eps=1e-7):
    """
    Solve the log barrier of nonnegative quadratic programming problem
        min. t(1/2 x'Px + q'x + r) - \sum \log x
    with newton method
    """

    def f(__x):
        return t * (0.5 * __x @ P @ __x + q @ __x + r) - np.sum(np.log(__x))

    def grad_hessian(__x):
        __g = t * (P @ __x + q) - 1./__x
        __h = t * P + np.diag(1/__x**2)
        return __g, __h

    if xstart is None:
        x = np.ones_like(q)
    else:
        x = xstart

    while True:
        fx = f(x)
        g, h = grad_hessian(x)
        dx = np.linalg.solve(h, -g)

        decrement = -1 * g @ dx
        if decrement <= eps:
            # print(f(x))
            return x, 1 / (-t * -x)

        # backtracking line search
        s = 1
        while np.min(x + s*dx) <= 0 \
        or f(x + s * dx) >= fx + ALPHA * s * g @ dx:
            s *= BETA

        x += s * dx

def nonnegative_quadratic_programming_interior_point_method(P, q, r, eps=1e-4):
    """
    Solve the nonnegative quadratic programming problem
        min. 1/2 * x'Px + q'x + r
        s.t. x >= 0
    with interior point method
    """

    m = len(q) # number of inequalities
    x = np.ones_like(q)

    t = 1
    while True:
        x, lamb = nonnegative_quadratic_programming_log_barrier(t, P, q, r, xstart=x)
        if m/t <= eps:
            return x, lamb
        t *= 2

def nonnegative_matrix_factorization(A, k, eps=1e-2, X_start=None, Y_start=None):
    """
    Solve the nonnegative matrix factorization problem
        min. ||A - XY||_F^2
        s.t. X_i,j >= 0
             Y_i,j >= 0
    with alternating convex optimization
    """
    def f(__X, __Y):
        return np.sum(
            (A - X @ Y) ** 2
        )

    m,n = A.shape
    X = np.random.rand(m,k) if X_start is None else X_start
    Y = np.random.rand(k,n) if Y_start is None else Y_start

    AAT = A @ A.T
    ATA = A.T @  A

    while True:

        # optimize over Y
        Y_new = Y.copy()
        ATX = A.T @ X
        XTX2 = 2 * X.T @ X
        for i in range(n):
            y, _ = nonnegative_quadratic_programming_interior_point_method(XTX2, -2*ATX[i,:], ATA[i,i])
            Y_new[:,i] = y

        y_change = np.linalg.norm(Y - Y_new)
        Y = Y_new

        # optimize over X
        X_new = X.copy()
        YAT = Y @ A.T
        YYT2 = 2 * Y @ Y.T
        for i in range(m):
            x, _ = nonnegative_quadratic_programming_interior_point_method(YYT2, -2*YAT[:,i], AAT[i,i])
            X_new[i,:] = x

        xchange = np.linalg.norm(X - X_new)
        X = X_new

        print(f(X, Y), xchange + y_change)

        if (xchange + y_change) < eps:
            return X, Y






if __name__ == '__main__':

    from scipy.io import savemat, loadmat

    m = 50
    n = 50
    k = 5

    # X = np.random.rand(m,k)
    # Y = np.random.rand(k,n)
    # A = X @ Y

    # savemat('mat_fact.mat', {'A':A})
    A = loadmat('mat_fact.mat')['A']

    nonnegative_matrix_factorization(A, k=k)






