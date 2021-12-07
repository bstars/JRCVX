"""
Solve the least-norm solution for linear equation with augmented lagrangian
"""

import numpy as np
from scipy.io import savemat, loadmat

def objective(x):
    return np.inner(x, x)

def lagrangian(x, A, b, v):
    return np.inner(x,x) + v @ (A @ x - b), 2 * x + A.T @ v

def augmented_lagrangian(x, A, b, v, c):
    return np.inner(x,x) + v @ (A @ x - b) + 0.5 * c * np.linalg.norm(A @ x - b), 2 * x + A.T @ v + c * A.T @ (A @ x - b)

def solve_augmented_lagrangian(x0, A, b, EPS=1e-2):
    m, n = A.shape
    c = 1
    x = x0
    v = np.zeros(shape=[m])
    I = np.eye(n)

    convergence = False

    while not convergence:

        xprime = np.linalg.solve(
            c * A.T @ A + 2 * I, A.T @ (c * b - v)
        )

        vprime = v + c * (A @ xprime - b)

        obj = objective(xprime)
        l, gradl = lagrangian(xprime, A, b, v)
        a, grada = augmented_lagrangian(xprime, A, b, vprime, c)

        print("obj:%.5f, L:%.5f, A:%.5f, gradl:%.5f, grada:%.5f, grad diff:%.5f" %
              (obj, l, a, np.linalg.norm(gradl), np.linalg.norm(grada),np.linalg.norm(gradl - grada))
              )

        if np.linalg.norm(xprime - x) + np.linalg.norm(vprime - v) <= EPS:
            return xprime, vprime



        x = xprime
        v = vprime
        # c *= 2





if __name__ == "__main__":
    m = 100
    n = 200


    # A = np.random.randn(m,n)
    # x0 = np.random.randn(n)
    # b = A @ x0
    # savemat("data.mat", {"A":A, "b":b, 'x0':x0})

    mdict = loadmat("data.mat")
    A = mdict['A']
    b = mdict['b'][0]
    x0 = mdict['x0'][0]
    x, v = solve_augmented_lagrangian(x0, A, b)

    print(x)
    print(np.linalg.norm(x))
