"""
This file solves the linear programming problem
    min. c'x
    s.t. x >= 0
         Ax = b
with primal-dual interior point method
"""

import numpy as np


def lp_primal_dual_interior_point_method(c:np.array, A:np.array, b:np.array, mu=10, ALPHA=0.01, BETA=0.5, eps_feas=1e-6, eps=1e-8):


    def residual(__x, __lamb, __v, __t):
        rp = A @ __x - b  # primal residual
        rd = c - __lamb + A.T @ __v # dual residual
        rc = __lamb * __x - 1/__t   # centrality residual

        ry = np.hstack([rd, rc, rp])
        return rd, rc, rp, ry

    def residual_norm(__x, __lamb, __v, __t):
        rd, rc, rp, ry = residual(__x, __lamb, __v, __t)
        return np.linalg.norm(ry)

    # initialization
    m,n = A.shape
    x = np.ones_like(c)
    lamb = 1. / x   # initial surrogate duality gap is number of inequalities
    v = np.zeros(shape=[m])


    while True:

        eta = x @ lamb  # surrogate duality gap \eta = -f(x)' * lamb
        print(eta)
        t = mu * n / eta

        rd, rc, rp, ry = residual(x, lamb, v, t)

        if np.linalg.norm(rp) < eps_feas and np.linalg.norm(rd) < eps and eta < eps:
            return x, lamb, v

        """ Solve the primal-dual search direction by block elimination """
        dv_lhs = - A @ np.diag(1 / lamb) @ np.diag(x) @ A.T
        dv_rhs = -rp + A @ np.diag(1/lamb) @ rc + A @ np.diag(1/lamb) @ np.diag(x) @ rd

        dv = np.linalg.solve(dv_lhs, dv_rhs)
        dlamb = rd + A.T @ dv
        dx = np.diag(1/lamb) @ (-rc - np.diag(x) @ dlamb)

        """ line search """
        s = 1
        while np.min(lamb + s * dlamb) <= 0:
            s *= BETA

        while np.min(x + s * dx) <= 0:
            s *= BETA

        r = np.linalg.norm(ry)
        while residual_norm(x + s * dx, lamb + s * dlamb, v + s * dv, t) >= (1 - ALPHA * s) * r:
            s *= BETA

        # print(s)
        x += s * dx
        v += s * dv
        lamb += s * dlamb





if __name__ == '__main__':
    m = 100
    n = 50

    # x = np.random.randn(n)
    # x = x * np.sign(x) + 0.001
    #
    # A = np.random.randn(m,n)
    # b = A @ x
    #
    # c = np.random.randn(n)

    from scipy.io import savemat, loadmat
    mdict = loadmat('lp.mat')
    A = mdict['A']
    b = mdict['b'][0]
    c = mdict['c'][0]



    x, lamb, v = lp_primal_dual_interior_point_method(c, A, b)

    print(c @ x)






