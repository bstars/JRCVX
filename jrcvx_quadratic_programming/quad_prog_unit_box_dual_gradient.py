"""
This file solves the quadratic programming no unit box

min. 1/2 x'Px + q'x + r
s.t. ||x||_inf <= 1

with projected subgradient method for dual function
"""

import numpy as np

def projectOnUnitBox(x):
    """ project x on unit box xi \in [-1,1]"""
    x_ = np.minimum(x, 1)
    x_ = np.maximum(x_,-1)
    return x_

def f(x, P, q):
    return 0.5 * x @ P @ x + q @ x

def fi(x):
    return x**2 - 1

def Lagrangian(x, P, q, lamb):
    return f(x, P, q) + x @ np.diag(lamb) @ x - np.sum(lamb)

def GradientOfLagrangian(P, q, x, lamb):
    return P @ x + q + 2 * np.diag(lamb) @ x

def quad_prog_sgm_dual(P, q, alpha=0.1, EPS = 1e-1):
    n = len(q)
    lamb = np.ones(shape=[n])
    f0s = []
    gs = []

    i = 1
    while True:
        lhs = P + 2 * np.diag(lamb)
        rhs = -q
        x_star_lamb = np.linalg.solve(lhs, rhs)
        x_star_lamb_projection = projectOnUnitBox(x_star_lamb)
        fx = f(x_star_lamb_projection, P, q)
        glamb = Lagrangian(x_star_lamb, P, q, lamb)

        f0s.append(fx)
        gs.append(glamb)
        gap = fx - glamb
        print("Iteration %d, duality gap %.4f" % (i, gap))
        i += 1

        if gap <= EPS:
            return x_star_lamb_projection, lamb, f0s, gs

        lamb = lamb + alpha * fi(x_star_lamb)
        lamb = np.maximum(lamb, 0)