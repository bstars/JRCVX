"""
Solve the analytic proble
    min. - \sum log(bi - ai*x)

with infeasible newton method
solve the equivalent problem

    min. - \sum log(yi)
    s.t. yi = bi - ai * x
"""

import numpy as np
import matplotlib.pyplot as plt

def ac_infeasible_start_newton_method(A:np.array, b:np.array, xstart:np.array=None, ALHPA=0.01, BETA=0.5, eps=1e-6):
    def construct_starting_point():
        m,n = A.shape
        x = np.random.randn(n)

        y = b - A @ x + 0.01
        idx = np.where(y<=0)
        y[idx] = 1

        return x, y

    def grad_hessian(__y):
        return -1./__y, np.diag(1./(__y**2))

    def residual_norm(__x:np.array, __y:np.array, __v:np.array):
        r = np.concatenate([
            A.T @ __v, -1./__y + __v, __y + A @ __x - b
        ])
        return np.linalg.norm(r, ord=2)

    if xstart is None:
        x, y = construct_starting_point()
    else:
        x = xstart
        y = b - A @ x

    m,n = A.shape
    v = np.zeros(shape=[m])

    while True:
        g, H = grad_hessian(y)
        rp = A @ x + y - b

        # solve for dx
        ATHA = A.T @ H @ A
        dx_rhs = A.T @ g - A.T @ H @ rp
        dx = np.linalg.solve(ATHA, dx_rhs)

        # solve for dy
        dy = -rp - A @ dx

        # solve for dv
        dv = -H @ dy - g - v

        rnorm = residual_norm(x, y, v)  # residual norm

        if rnorm <= eps:
            return x, y, v

        t = 1
        while np.min(y + t * dy) <=0 or residual_norm(x+t*dx, y+t*dy, v+t*dv) > (1-ALHPA*t) * rnorm:
            t *= BETA

        #print(t, rnorm)
        x += t * dx
        y += t * dy
        v += t * dv


def plot_inequalities(Ab:np.array, xmin=-1, xmax=3):
    """ Each row represents ax1 + bx2 < c """
    x1range = np.arange(xmin, xmax, 0.1)

    for a, b, c in Ab:
        x2range = -float(a)/b * x1range + float(c)/b
        plt.plot(x1range, x2range)

if __name__ == "__main__":
    Ab = np.array([
        [1, 1, 2],      # x1 + x2 <= 2
        [-1, 1, 0],      # x2 < x1
        [0, -1, 0]      # x2 > 0
    ])

    A = Ab[:,:2]
    b = Ab[:,2]
    x, y, v = ac_infeasible_start_newton_method(A, b)
    plot_inequalities(Ab)
    plt.plot(x[0], x[1], 'ro')
    plt.show()



