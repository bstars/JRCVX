"""
Solve a piece-wise linear minimization
    min. max_i ai*x + bi
with analytic center cutting plane method
"""

import sys
sys.path.append('..')

import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

from jrcvx_analytic_center.analytic_center_infeasible_newton import ac_infeasible_start_newton_method
from jrcvx_piece_wise_linear.pwl_linear_programming import pwl_interior_point_method

def pwl_cutting_plane_method(A:np.array, b:np.array, C:np.array, d:np.array, orcale_type='neutral', eps=5e-4):
    """
    Solve a piece-wise linear minimization
        min. max_i ai*x + bi
    with analytic center cutting plane method

    :param C:
    :param d:
        These parameter indicates the initial localization polyhedron {x|Cx<=d}
    """
    def f_subgrad(__x):
        val = A @ __x + b
        idx = np.argmax(val)
        return val[idx], A[idx,:]

    m, n = A.shape
    x = np.ones(shape=[n])

    fbest = None
    num_iter = 0

    while True:
        num_iter += 1
        x_prev = x.copy()
        x, _, _ = ac_infeasible_start_newton_method(C, d)
        newf, g = f_subgrad(x)

        if fbest is None or newf<=fbest:
            fbest = newf

        print(C.shape)

        if np.linalg.norm(x - x_prev) <= eps:
            return x, f_subgrad(x)[0]

        if orcale_type is 'neutral':
            # a neutral cut
            # f(z) >= f(x) + g @ (z-x)
            # g @ (z-x) >= 0     ---->     f(z) >= f(x)
            C = np.concatenate([C, np.array([g])], axis=0)
            d = np.append(d, g @ x)
        else:
            # a deep cut
            # g@z <= f_best^k + g @ x - f(x)
            C = np.concatenate([C, np.array([g])], axis=0)
            d = np.append(d, fbest + g @ x - newf)



if __name__ == "__main__":
    mdict = loadmat("data.mat")
    A = mdict['A']
    b = mdict['b'][0]
    m,n = A.shape
    C = np.concatenate(
        [np.eye(n), -1 * np.eye(n)], axis=0
    )
    d = np.ones(shape=[2 * n])
    _, val_lp = pwl_interior_point_method(A, b)
    x, val_accpm = pwl_cutting_plane_method(A,b,C,d)

    print("cutting plane method: %.7f, linear programming:%.7f" % (val_accpm, val_lp))

