"""
Solve the analytic center problem
    min. - \sum \log(b-Ax)

with feasible start newton method
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from jrcvx_piece_wise_linear.pwl_linear_programming import pwl_interior_point_method

def ac_feasible_start_newton_method(A:np.array, b:np.array, xstart=None, ALPHA=0.01, BETA=0.5, eps=1e-6):
    """
    Solve the analytic center problem
        min. - \sum \log(b - Ax)
    """
    if xstart is not None:
        x = xstart
        feasible = True
    else:
        x, t = pwl_interior_point_method(A, -b)
        feasible = True if t < 0 else False
    if not feasible:
        raise Exception('Analytic center problem is in feasible')

    def f(__x):
        return -np.sum(np.log(b - A @ __x + 1e-8))

    def grad_hessian(__x):
        slackness = b - A @ __x + 1e-8
        __g = A.T @ (1 / slackness)
        __h = A.T @ np.diag(1 / slackness**2) @ A
        return  __g, __h

    while True:
        val = f(x)
        g, h = grad_hessian(x)
        dx = np.linalg.solve(h, -g)
        decrement = -1 * g @ dx

        if decrement <= eps:
            return x

        # backtracking line search
        s = 1
        while np.min(b - A @ (x + s * dx)) <=0 or f(x + s * dx) > val + ALPHA * s * np.inner(g, dx):
            s *= BETA
        x += s * dx


def plot_inequalities(Ab:np.array, xmin=-1, xmax=3):
    """ Each row represents ax1 + bx2 < c """
    x1range = np.arange(xmin, xmax, 0.1)

    for a, b, c in Ab:
        x2range = -float(a)/b * x1range + float(c)/b
        plt.plot(x1range, x2range)

if __name__ == "__main__":

    Ab = np.array([
        [1, 1, 2],  # x1 + x2 <= 2
        [-1, 1, 0],  # x2 < x1
        [0, -1, 0]  # x2 > 0
    ])

    A = Ab[:, :2]
    b = Ab[:, 2]
    x = ac_feasible_start_newton_method(A,b)
    plot_inequalities(Ab)
    plt.plot(x[0], x[1], 'ro')
    plt.show()