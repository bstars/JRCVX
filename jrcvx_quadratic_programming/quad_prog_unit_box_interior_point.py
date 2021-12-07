"""
This file solves the quadratic programming no unit box

min. 1/2 x'Px + q'x + r
s.t. ||x||_inf <= 1

with interior-point method
"""

import sys
sys.path.append('..')
import numpy as np

from jrcvx_quadratic_programming.quad_prog_unit_box_dual_gradient import quad_prog_sgm_dual



def f(P, q, x):
    return 0.5 * x @ P @ x + q @ x

def generateRandomSymmetrixMatrix(n):
    A = np.random.randn(n,n)
    A = A.T @ A
    return A

def generateRandomPSDMatrix(n, definite=False):
    X = generateRandomSymmetrixMatrix(n)
    evals, evecs = np.linalg.eig(X)
    eps = 1 if definite else 0
    evals = np.maximum(evals, eps+0)
    return evecs @ np.diag(evals) @ evecs.T

def quad_prog_barrier(t, P, q, r,xstart=None, eps=1e-8, ALPHA=0.01, BETA=0.5):
    """
    Solve the log-barrier of unit-box-constrained quadratic programming
        min. t * (1/2 x'Px + q'x + r) - \sum \log (1-x)

    """

    def grad_hessian(__x):
        __g = t * (P @ __x + q) + 2 * __x / (1 - __x**2)
        __h = t * P + np.diag(
            (2 + 2 * __x**2) / (1-__x**2)**2
        )
        return __g, __h


    def val(__x):
        return t * (0.5 * __x @ P @ __x  + q @ __x + r) - np.sum(np.log(1-__x**2))

    if xstart is None:
        x = np.zeros_like(q)
    else:
        x = xstart

    while True:
        fx = val(x)
        g, h = grad_hessian(x)
        dx = np.linalg.solve(h,  -g)

        decrement = -1 * g @ dx
        # print(decrement)

        if decrement <= eps:
            # print("grad norm",np.linalg.norm(g))
            return x, -1/t * 1 / (1 - x**2)   # primal variable and dual variable


        s = 1
        while np.min(1 - (x + s * dx)**2) <= 0 \
        or val(x + s * dx) > fx + ALPHA * s * g @ dx:
            s *= BETA

        x += s * dx

def quad_prog_interior_point_method(P, q, r, eps=1e-5):
    """
    Solve the unit-box-constrained quadratic programming
        min. t * (1/2 x'Px + q'x + r)
        s.t. xi^2 <= 1
    """
    t = 1
    x = np.zeros_like(q)
    m = len(q)  # num of inequality constraints

    while m/t >= eps:
        x, lamb = quad_prog_barrier(t, P, q, r, x)
        t *= 2
    return x, lamb





if __name__ == '__main__':
    n = 500
    P = generateRandomPSDMatrix(n)
    q = np.random.randn(n)
    x = np.random.randn(n)
    r = 1

    from scipy.io import savemat, loadmat
    mdict = loadmat('data.mat')
    P = mdict['P']
    q = mdict['q'][0]

    print(P.shape, q.shape)



    x1, _ = quad_prog_interior_point_method(P, q, r)
    x2, _, _, _ = quad_prog_sgm_dual(P, q, EPS=1e-5)

    print(
        f(P, q, x1), f(P, q, x2)
    )








