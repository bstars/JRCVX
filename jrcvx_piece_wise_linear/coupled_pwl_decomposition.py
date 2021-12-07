"""
Solve the coupled piece-wise linear minimization problem
    min(x1, x2, y). max(A1 [x1, y] + b1) + max(A2 [x2, y] + b2)
where y is a scalar coupling/complicating variable.

To demonstrate the usage of bisection in solving the master problem,
this file only deals with one coupling variable,
but only minor change is required to extend these methods to deal with multiple coupling variables.
"""

import sys
sys.path.append('..')

import numpy as np
from scipy.io import loadmat, savemat

from jrcvx_piece_wise_linear.variable_constrained_pwl_linear_programming import variable_constrained_pwl_interior_point_method
from jrcvx_piece_wise_linear.pwl_linear_programming import pwl_interior_point_method


def generate_data():
    m = 50
    n = 15

    A1 = np.random.randn(m, n)
    A2 = np.random.randn(m, n)
    b1 = np.random.randn(m)
    b2 = np.random.randn(m)

    savemat(
        'data_coupled.mat',
        mdict={
            'A1':A1,
            'A2':A2,
            'b1':b1,
            'b2':b2
        }
    )

def coupled_pwl_decomposition_method_primal_decomposition(A1:np.array, A2:np.array, b1:np.array, b2:np.array, eps=1e-5):
    """
    Solve the coupled piece-wise linear minimization problem
        min(x1, x2, y). max(A1 [x1, y] + b1) + max(A2 [x2, y] + b2)
    where y is a scalar coupling/complicating variable

    The subproblems
        min(x1). max(A1 [x1, y] + b1)
        min(x2). max(A2 [x2, y] + b2)
    are solved by linear programming.

    The master problem
        min(y). phi1(y) + phi2(y)
    is solving by bisection since y is a scalar.
    """
    m,n = A1.shape
    num_private = n - 1
    x1 = np.zeros(shape=[num_private])
    x2 = np.zeros(shape=[num_private])
    y = np.zeros(shape=[1])


    yl = np.ones(shape=[1]) * -20
    yr = np.ones(shape=[1]) * 20


    while True:

        y = (yl + yr) / 2
        xy1, f1, v1 = variable_constrained_pwl_interior_point_method(A1, b1, y=y)
        xy2, f2, v2 = variable_constrained_pwl_interior_point_method(A2, b2, y=y)

        if (yr - yl) <= eps:
            return xy1[:-1], xy2[:-1], y, f1 + f2
        # print(f1 + f2)

        if (v1 + v2) > 0:
            yl = y
        else:
            yr = y

def coupled_pwl_decomposition_method_dual_decomposition(A1:np.array, A2:np.array, b1:np.array, b2:np.array, eps=1e-5):
    """
    Solve the coupled piece-wise linear minimization problem
        min(x1, x2, y). max(A1 [x1, y] + b1) + max(A2 [x2, y] + b2)
    where y is a scalar coupling/complicating variable

    The subproblems
        min(x1,y1). max(A[x1,y1] + b1) + v.T @ y1
        min(x2,y2). max(A[x1,y2] + b2) - v.T @ y2
    are solved by linear programming

    The master problem
        max(v). g1(v) + g2(v)
    is solved by bisection.
    """
    vl = np.ones(shape=[1]) * -20
    vr = np.ones(shape=[1]) * 20



    while True:
        v = (vl + vr) / 2

        A1_temp = A1.copy()
        A1_temp[:,-1] += v
        x1y1, f1 = pwl_interior_point_method(A1_temp, b1)
        x1, y1 = x1y1[:-1], x1y1[-1]


        A2_temp = A2.copy()
        A2_temp[:,-1] -= v
        x2y2, f2 = pwl_interior_point_method(A2_temp, b2)
        x2, y2 = x2y2[:-1], x2y2[-1]

        # print(f1 + f2)

        if np.abs(y2 - y1) <= eps:
            y = (y1 + y2) / 2
            xy1, f1, v1 = variable_constrained_pwl_interior_point_method(A1, b1, y=y)
            xy2, f2, v2 = variable_constrained_pwl_interior_point_method(A2, b2, y=y)
            # print(f1 + f2)
            return xy1[:-1], xy2[:-1], y, f1 + f2

        if (y2 - y1) > 0:
            vr = v
        else:
            vl = v




if __name__ == '__main__':
    mdict = loadmat('data_coupled.mat')
    A1 = mdict['A1']
    A2 = mdict['A2']
    b1 = mdict['b1'][0]
    b2 = mdict['b2'][0]

    x1, x2, y, val1 = coupled_pwl_decomposition_method_primal_decomposition(A1, A2, b1, b2)
    x1, x2, y, val2 = coupled_pwl_decomposition_method_dual_decomposition(A1, A2, b1, b2)

    print(val1, val2)


