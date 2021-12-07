"""
This file solves the linear semi-definite programming
    min. c'x
    s.t. (\sum Fi * xi) + G <= 0
         Ax = b

with interior point method
"""


import sys
sys.path.append('..')

import numpy as np
from utils.random_generate import generate_random_symmetric_matrix, generate_random


def linear_sdp_log_barrier(t, c, A, b, F, G, xstart=None, ALPHA=0.01, BETA=0.5, eps=1e-7):
    """
    Solves the log barrier of linear semidefinite programming
        min. t * c'x - logdet(-F(x))
        s.t. Ax = b

        F(x) = (\sum xi * Fi)+ G

    c \in R^n
    A \in R^{mxn}
    b \in R^{m}
    F \in R^{pxpxn} are n pxp symmetrix matrices
    G \in R^{pxp}
    """

    n,n,p = F.shape
    m, n = A.shape

    def F_func(__x):
        return np.sum(__x * F, axis=-1) + G

    def f_func(__x):
        sign, ld = np.linalg.slogdet(-1 * F_func(__x))
        return t * c @ __x - sign * ld

    def FiFinv_func(__x):
        # return a matrix of nxpxp
        ret = []
        Fxinv = np.linalg.inv(F_func(__x))
        for i in range(n):
            ret.append(
                F[:,:,i] @ Fxinv
            )
        return np.stack(ret)

    def gradient(__x, __FiFinv):
        g = np.zeros([n])
        for i in range(n):
            g[i] = t * c[i] - np.trace(__FiFinv[i])
        return g

    def construct_TR(__FiFinv):
        TR = np.zeros(shape=[n,n])
        for i in range(n):
            for j in range(n):
                TR[i,j] = np.trace(__FiFinv[i] @ __FiFinv[j])
        return TR

    def construct_rhs(__FiFinv):
        rhs = np.zeros(shape=[n])
        for i in range(n):
            rhs[i] = np.trace(__FiFinv[i])
        rhs -= t * c

        ret = np.zeros(shape=[m+n])
        ret[:n] = rhs
        return ret

    x = xstart.copy()
    num = 0
    while True:
        FiFinv = FiFinv_func(x)

        # construct the KKT system
        TR = construct_TR(FiFinv)
        rhs = construct_rhs(FiFinv)

        lhs = np.zeros(shape=[m+n, m+n])
        lhs[:n,:n] = TR
        lhs[:n,n:] = A.T
        lhs[n:,:n] = A

        dxv = np.linalg.solve(lhs, rhs)
        dx = dxv[:n]
        v = dxv[n:]

        g = gradient(x, FiFinv)

        # stopping criterion : newton decrement
        if -1 * g @ dx <= eps:
            # print(f_func(x))
            return x, v, np.linalg.inv(F_func(x)), num

        # back-tracking line search
        s = 1
        evals, evecs = np.linalg.eig(F_func(x + s * dx))
        maxeval = np.max(evals)
        while maxeval >= 0:
            s *= BETA
            evals, evecs = np.linalg.eig(F_func(x + s * dx))
            maxeval = np.max(evals)

        fx = f_func(x)
        while (fx + s * ALPHA * g @ dx) < f_func(x + s * dx):
            s *= BETA

        x += s * dx
        num += 1

def linear_sdp_interior_point_method(c, A, b, F, G, xstart, eps=1e-6):
    x = xstart
    p, _, n = F.shape
    t = 1
    while True:
        x, v, Finv, num = linear_sdp_log_barrier(t, c, A, b, F, G, x)
        print('t=%d, %d newton steps' % (t, num))
        if (p / t) < eps:
            print('Optimal value: %.6f, Dualtity gap: %.10f'%(c @ x, p/t))
            return x, -1/t * Finv, v / t    # optimial primal variable and dual variable
        t *= 5

def generate_data():
    m = 10
    n = 20
    p = 20
    x = generate_random(n)
    A = generate_random(m,n)
    A[-1,:] = np.ones(n)    # make the sub-level set bounded
    b = A @ x
    c = generate_random(n)


    F = []
    for i in range(n):
        Fi = generate_random_symmetric_matrix(p)
        F.append(Fi)

    F = np.stack(F, axis=-1)
    xF = np.sum(x * F, axis=-1)
    G = -np.eye(p) * 5 - xF # extra slackness


    from scipy.io import savemat
    savemat(
        './data.mat',
        mdict={
            'A':A,
            'b':b,
            'F':F,
            'G':G,
            'c':c,
            'xstart' : x
        }
    )




if __name__ == "__main__":
    # generate_data()
    from scipy.io import loadmat
    mdict = loadmat('./data.mat')


    A = mdict['A']
    b = mdict['b'][0]
    c = mdict['c'][0]
    F = mdict['F']
    G = mdict['G']
    xstart = mdict['xstart'][0]

    # linear_sdp_log_barrier(100, c, A, b, F, G, xstart)
    x, lamb, v = linear_sdp_interior_point_method(c, A, b, F, G, xstart)














