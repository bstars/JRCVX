"""
Solve the variable-constrained piece-wise linear minimization problem
    min(x,z). max(A[x,z] + b)
    s.t.      z = y

Though this problem seems to be trivial at first glance,
the dual variable gives the local sensitivity to variable constraint,
which has significant importance in solving master problem in coupled pwl minimization.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.io import loadmat

from jrcvx_piece_wise_linear.pwl_linear_programming import pwl_interior_point_method


def variable_constrained_pwl_barrier(c, A:np.array, b:np.array, y:float, start_point:np.array=None, eps=1e-8, ALPHA=0.01, BETA=0.5):
    """
    Solve log-barrier the variable-constrained piece-wise linear minimization
        min(x,z,t). c * t - \sum \log (t1 - A[x,z] - b)
        s.t.        z = y
    This method only deal with one variable constraint (i.e. y is a scalar)
    """
    m, n = A.shape

    zzc = np.zeros(shape=[n+1]) # zero-zero-c
    zzc[-1] = c

    zoz = np.zeros(shape=[n+1]) # zero-one-zero
    zoz[-2] = 1

    mA1 = np.hstack([-A, np.ones(shape=[m,1])])

    def f(__xzt):
        slackness = mA1 @ __xzt - b + 1e-8
        return zzc @ __xzt - np.sum(np.log(slackness))

    def grad_hessian_inv(__xzt):
        slackness = mA1 @ __xzt - b + 1e-8
        __g = zzc - mA1.T @ (1. / slackness)
        __h = mA1.T @ np.diag(1 / (slackness**2)) @ mA1
        return __g, __h

    if start_point is None:
        # construct a feasible point
        xzt = np.zeros(shape=[n+1])
        xzt[-2] = y
        slackness = b - mA1 @ xzt
        xzt[-1] = np.max(slackness) + 0.1 # extra slackness

    else:
        xzt = start_point

    num_iter = 0
    while True:
        num_iter += 1
        val = f(xzt)
        g, h = grad_hessian_inv(xzt)
        h_inv = np.linalg.pinv(h)

        # solve the KKT system with block elimination
        # solve for dual variable v
        v_lhs = -1 * zoz @ h_inv @ zoz
        v_rhs = zoz @ h_inv @ g
        v = v_rhs / v_lhs

        # solve for dx dz dt
        dxzt = h_inv @ (-1 * g - v * zoz)
        # print(dxzt[-2])

        decrement = -1 * g @ dxzt
        #print(decrement)

        if decrement < eps:
            return xzt, v

        # backtracking line search
        s = 1
        while np.min(mA1 @ (xzt + s * dxzt) - b) <= 0:
            s *= BETA

        while f(xzt + s * dxzt) > val + ALPHA * s * np.inner(g, dxzt):
            s *= BETA

        xzt += s * dxzt


def variable_constrained_pwl_interior_point_method(A, b, y, eps=1e-6):
    """
    Solve the variable constrained piece-wise linear problem
        min(x,z). max(A[x,z] + b)
        s.t.      z = y

    by solving the equivalent problem
        min(x,z). t
        s.t.      A[x,z] + b <= t1
                  z = y
    with interior point method
    """
    c = 1
    m,n = A.shape
    xzt = np.zeros(shape=[n + 1])
    xzt[-2] = y
    mA1 = np.hstack([-A, np.ones(shape=[m, 1])])
    slackness = b - mA1 @ xzt
    xzt[-1] = np.max(slackness) + 0.1  # extra slackness

    # xz, t, v = variable_constrained_pwl_barrier(c, A, b, y, xzt)

    while True:
        xzt,v = variable_constrained_pwl_barrier(c, A, b, y, start_point=xzt)

        if m / c < eps:
            return xzt[:-1], xzt[-1], v/c
        c *= 2


if __name__ == "__main__":
    mdict = loadmat('data.mat')
    A = mdict['A']
    b = mdict['b'][0]

    # variable_constrained_pwl_barrier(1, A, b, 1)

    xz, t, v = variable_constrained_pwl_interior_point_method(A, b, y=1)
    print(t)

    xz, t = pwl_interior_point_method(A[:,:-1], b + 1 * A[:,-1])
    print(t)







