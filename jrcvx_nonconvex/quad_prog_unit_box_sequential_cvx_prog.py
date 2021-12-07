"""
This file solves the non-convex quadratic programming on unit box

    min. 1/2 x'Px + q'x + r     (P is symmetric but not psd)
    s.t. ||x||_inf <= 1

with sequential convex programming
"""

import sys
sys.path.append('..')

import numpy as np


def trust_region_constrained_cvx_quad_prog_log_barrier(t, P, q, r, xk, rou, xstart=None, ALPHA=0.01, BETA=0.5, eps=1e-7):
    """
    Solves the log-barrier of trust region constrained convex quadratic programming over unit box
        min. t (1/2 x'Px + q'x + r)- log(x+1) - log(1-x) - log(x-a) - log(b-x)
    where a = xk - rou, b = xk + rou
    """

    a = xk - rou
    b = xk + rou

    if xstart is None:
        x = np.zeros_like(q)
        x = np.maximum(x, -1 + 0.01)   # x >= -1
        x = np.minimum(x, 1 - 0.01)    # x <= 1
        x = np.maximum(x, a + 0.01)  # x >= a
        x = np.minimum(x, b - 0.01)
    else:
        x = xstart

    def f(__x):
        return t * (0.5 * __x @ P @ __x + q @ __x + r) - np.sum(np.log(__x+1)) - np.sum(np.log(1-__x)) - np.sum(np.log(__x-a)) - np.sum(np.log(b-__x))

    def grad_hessian(__x):
        __g = t * (P @ __x + q) - 1/(__x+1) + 1/(1-__x) - 1/(__x-a) + 1/(b-__x)
        __h = t * P + np.diag(1/(__x+1)**2) + np.diag(1/(1-__x)**2) + np.diag(1/(__x-a)**2) + np.diag(1/(b-__x)**2)
        return __g, __h

    while True:
        fx = f(x)

        g, h = grad_hessian(x)
        dx = np.linalg.solve(h, -g)

        decrement = -1 * g @ dx
        # print(decrement)
        if decrement <= eps:
            # print(f(x))
            return x, 1 / (-t * (-x-1)), 1/(-t * (x-1)), 1/(-t*(-x+1)), 1/(-t*(x-b))    # primal variable and 4 dual variables

        # backtracking line search
        s = 1
        while np.min(x + s*dx + 1) <= 0 \
        or np.min(1 - x - s*dx) <= 0 \
        or np.min(x + s*dx - a) <= 0 \
        or np.min(b - x - s*dx) <= 0 \
        or f(x + s*dx) >= fx + ALPHA * s * g @ dx:
            s *= BETA
        x += s * dx

def trust_region_constrained_cvx_quad_prog_interior_point(P, q, r, xk, rou, xstart=None, eps=1e-4):

    """
    Solve the trust region constrained quadratic programming over unit box
        min. 1/2 x'Px + q'x + r
        s.t. ||x||_inf <= 1
             ||x-xk|| <= rou
    :param P:
    :param q:
    :param r:
    :param xstart:
    :return:
    """

    a = xk - rou
    b = xk + rou

    if xstart is None:
        x = np.zeros_like(q)
        x = np.maximum(x, -1 + 0.01)  # x >= -1
        x = np.minimum(x, 1 - 0.01)  # x <= 1
        x = np.maximum(x, a + 0.01)  # x >= a
        x = np.minimum(x, b - 0.01)  # x <= b
    else:
        x = xstart

    m = 4 * len(q)
    t = 1
    while True:
        x, lamb1, lamb2, lamb3, lamb4 = trust_region_constrained_cvx_quad_prog_log_barrier(t, P, q, r, xk, rou)
        if m/t <= eps:
            return x, lamb1, lamb2, lamb3, lamb4
        t *= 2

def non_convex_quad_prog_sequential_programming(P, q, r, xstart=None, ALPHA=0.3, beta_succ=1.1, beta_fail=0.5):
    """
     Solve the non-convex quadratic programming over unit box
        min. 1/2 x'Px + q'x + r
        s.t. ||x||_inf <= 1

    with sequential convex programming
    """
    rou = 0.05
    if xstart is None:
        x = np.zeros_like(q)
    else:
        x = xstart

    def quad_form(P, q, r, __x):
        return 0.5 * __x @ P @ __x + q @ __x + r

    # project on psd
    lamb, U = np.linalg.eig(P)
    P_bar = U @ np.diag(np.maximum(lamb, 0)) @ U.T

    x_update = True
    fx = None

    history = []

    while True:

        if x_update:
            fx = quad_form(P, q, r, x)
            q_bar = P @ x + q - P_bar @ x
            r_bar = quad_form(P_bar, -P @ x - q, fx, x)
        print(fx)
        history.append(fx)

        x_new, _, _, _, _ = trust_region_constrained_cvx_quad_prog_interior_point(P_bar, q_bar, r_bar, x, rou, x)

        predicted_decrease = quad_form(P_bar, q_bar, r_bar, x) - quad_form(P_bar, q_bar, r_bar, x_new)
        actual_decrease = quad_form(P, q, r, x) - quad_form(P, q, r, x_new)

        if np.linalg.norm(x - x_new) <= 1e-3 * np.linalg.norm(x):
            return x_new, history

        if actual_decrease >= ALPHA * predicted_decrease:
            rou = beta_succ * rou
            x = x_new
            x_update = True
        else:
            rou = beta_fail * rou
            x_update = False




if __name__ == "__main__":


    def trust_region_test():
        from utils.random_generate import generate_random_symmetric_matrix, generate_random_psd_matrix, generate_random_vector
        from scipy.io import savemat, loadmat

        mdict = loadmat('barrier.mat')
        P = mdict['P']
        q = mdict['q'][0]
        xk = mdict['xk'][0]

        r = 1
        rou = 0.01
        t = 1

        # x, lamb1, lamb2, lamb3, lamb4 = trust_region_constrained_cvx_quad_prog_log_barrier(1, P, q, r, xk, rou)
        x, lamb1, lamb2, lamb3, lamb4 = trust_region_constrained_cvx_quad_prog_interior_point(P, q, r, xk, rou)
        print(x)
        print(lamb1)
        print(lamb2)
        print(lamb3)
        print(lamb4)

        print(0.5 * x @ P @ x + q @ x + r)


    def noncvx_qp_test():
        n = 20
        from utils.random_generate import generate_random_symmetric_matrix, generate_random_vector
        from scipy.io import savemat, loadmat
        import matplotlib.pyplot as plt
        # P = generate_random_symmetric_matrix(n)
        # q = generate_random_vector(n)
        # r = 0
        # savemat(
        #     'noncvx_qp.mat',
        #     mdict={
        #         'P' : P,
        #         'q' : q
        #     }
        # )

        mdict = loadmat('noncvx_qp.mat')
        P = mdict['P']
        q = mdict['q'][0]
        r = 1.

        n = len(q)
        for i in range(5):
            x = (np.random.rand(n) * 2 - 1) * 0.5
            x, history = non_convex_quad_prog_sequential_programming(P, q, r, xstart=x)
            plt.plot(history, label='trial %d' % (i))
        plt.legend()
        plt.show()
    noncvx_qp_test()










