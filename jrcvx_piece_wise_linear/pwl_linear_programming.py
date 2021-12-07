"""
Solve the Piece-Wise Linear problem
        min. max(Ax + b)
"""

import numpy as np


def pwl_barrier(A:np.array, b:np.array, c, xstart:np.array=None, eps=1e-8, ALPHA=0.01, BETA=0.5):
    """
    Solve the problem
        min. c * t - \sum \log(t1 - Ax - b)

    """
    m, n = A.shape
    zeroc = np.zeros(shape=[n + 1])
    zeroc[-1] = 1
    zeroc *= c

    mA1 = np.hstack([-A, np.ones(shape=[m,1])])

    def f(__xt):
        slackness = mA1 @ __xt - b + 1e-8
        return zeroc @ __xt - np.sum(np.log(slackness))

    def grad_hessian(__xt):
        slackness = mA1 @ __xt - b + 1e-8
        __g = zeroc - mA1.T @ (1 / slackness)
        __h = mA1.T @ np.diag(1 / (slackness**2)) @ mA1
        return __g, __h

    xt = xstart.copy()

    num_iter = 0
    while True:
        num_iter += 1
        val = f(xt)
        g, h = grad_hessian(xt)

        # solve the KKT system
        dxt = np.linalg.solve(h, -g)
        decrement = -1 * g @ dxt

        if decrement < eps:
            return xt

        # backtracking line search
        s = 1
        while f(xt + s * dxt) > val + ALPHA * s * np.inner(dxt, g) \
            or np.min(mA1 @ (xt + s * dxt) - b)<=0:
            s *= BETA

        xt += s * dxt


def pwl_interior_point_method(A, b, eps=1e-6):
    """
    Solve the Piece-Wise Linear problem
        min. max(Ax + b)

    by solving the equivalent problem
        min. t
        s.t. Ax + b <= t1

    with interior-point method

    :return:
    """
    c = 1
    m,n = A.shape

    xt = np.zeros(shape=[n+1])
    xt[-1] = np.max(b) + 0.1    # 0.1 is some extra slackness to make sure it's strictly feasible for log barrier

    num_iter = 0
    while True:
        num_iter += 1
        xt = pwl_barrier(A, b, c, xstart=xt)
        if m / c < eps:
            return xt[:-1], xt[-1]
        c *= 2


if __name__ == '__main__':
    m = 100
    n = 20

    A = np.random.randn(m,n)
    b = np.random.randn(m)

    x, t = pwl_interior_point_method(A, b)
    print(t, np.max(A @ x + b))
