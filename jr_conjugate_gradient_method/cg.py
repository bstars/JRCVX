import sys

import matplotlib.pyplot as plt

sys.path.append('../..')

import numpy as np
import time

from JRCVX.utils.random_generate import generate_random_psd_matrix
from JRCVX.utils.plot_ellipsoid import plot_2d_ellipsoid, show


U = np.array([[2,-1],[1,2]])
U = U / np.linalg.norm(U,axis=1)
lamb = np.diag([1/4, 1/25])

u = np.array([2,1])

P = U @ lamb @ U.T

A = 2 * P
b = 2 * P @ u
c = u @ P @ u

A_half = np.sqrt(2) * U @ lamb**0.5 @ U.T


def f(x):
    return 1/2 * x.T @ A @ x - b @ x + c

def r(x):
    return b - A @ x


# plot_2d_ellipsoid(u, P, levels=[0.5, 1, 1.5, 2, 2.5])
# show()

x_start = np.array([2,6]).astype(float)
x_start = np.array([-2,4]).astype(float)

def steepest_descent():
    x = x_start.copy()

    xs = [x.copy()]
    while True:
        ri = r(x)
        if np.linalg.norm(ri) <= 1e-6:
            break
        alpha_i = (ri @ ri) / (ri @ A @ ri)
        x += alpha_i * ri
        xs.append(x.copy())

    plot_2d_ellipsoid(u, P, levels=[0.5, 1, 1.5, 2, 2.5])
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], 'r--')
    show()

# steepest_descent()


# chang of coordinate
# levels = np.array([0.5, 1, 1.5, 2, 2.5])
# plot_2d_ellipsoid(u, P, levels=levels)
# show()
# levels_u = levels + (b @ np.linalg.inv(A) @ b) / 2 -c
# levels_u *= 2
# plot_2d_ellipsoid(np.linalg.inv(A_half) @ b, np.eye(2), levels=levels_u)
# show()



# steepest descent with conjugate directions
def steepest_descent_with_conjugate_direction():
    x = x_start.copy()
    di = np.array([1.,0])

    ri = r(x)
    alpha = (ri.T @ di) / (di.T @ A @ di)
    new_x = x + alpha * di

    xs = np.stack([x, new_x, u])
    us = (A_half @ xs.T).T


    levels = np.array([0.5, 1, 1.5, 2, 2.5])
    plot_2d_ellipsoid(u, P, levels=levels)
    plt.plot(xs[:, 0], xs[:, 1], 'r--')
    show()


    levels_u = levels + (b @ np.linalg.inv(A) @ b) / 2 -c
    levels_u *= 2
    plot_2d_ellipsoid(np.linalg.inv(A_half) @ b, np.eye(2), levels=levels_u)
    plt.plot(us[:, 0], us[:, 1], 'r--')
    show()

# steepest_descent_with_conjugate_direction()

def almost_conjugate_gradient_method():
    x = x_start.copy()
    xs = [x.copy()]
    ds = []
    while True:
        ri = r(x)
        if np.linalg.norm(ri) <= 1e-6:
            break
        di = ri.copy()
        for dj in ds:
            di -= (dj @ A @ ri) / (dj @ A @ dj) * dj
        alphai = (ri @ di) / (di @ A @ di)
        x += alphai * di
        xs.append(x.copy())
        ds.append(di)

    xs = np.stack(xs)
    us = (A_half @ xs.T).T

    levels = np.array([0.5, 1, 1.5, 2, 2.5])
    plot_2d_ellipsoid(u, P, levels=levels)
    plt.plot(xs[:, 0], xs[:, 1], 'r-')
    show()

    levels_u = levels + (b @ np.linalg.inv(A) @ b) / 2 - c
    levels_u *= 2
    plot_2d_ellipsoid(np.linalg.inv(A_half) @ b, np.eye(2), levels=levels_u)
    plt.plot(us[:, 0], us[:, 1], 'r--')
    show()
# almost_conjugate_gradient_method()


def conjugate_gradient_method_eg():
    x = x_start.copy()
    d_prev = None
    r_prev = None
    xs = [x.copy()]

    num_iter = 0
    while True:
        ri = r(x)
        if np.linalg.norm(ri) <= 1e-6:
            break
        riri = ri @ ri
        if num_iter == 0:
            di = ri
        else:
            di = ri + riri / (r_prev @ r_prev) * d_prev
        alphai = riri / (di @ A @ di)
        x += alphai * di

        num_iter += 1
        r_prev = ri
        d_prev = di
        xs.append(x.copy())

    xs = np.stack(xs)
    us = (A_half @ xs.T).T

    levels = np.array([0.5, 1, 1.5, 2, 2.5])
    plot_2d_ellipsoid(u, P, levels=levels)
    plt.plot(xs[:, 0], xs[:, 1], 'r--')
    show()

    levels_u = levels + (b @ np.linalg.inv(A) @ b) / 2 - c
    levels_u *= 2
    plot_2d_ellipsoid(np.linalg.inv(A_half) @ b, np.eye(2), levels=levels_u)
    plt.plot(us[:, 0], us[:, 1], 'r--')
    show()

# conjugate_gradient_method_eg()


def conjugate_gradient_method(A, b):
    """Conjugate Gradient Method to solve the equation Ax = b with positive definite A"""
    x = np.zeros_like(b)
    def residual(__x):
        return b - A @ __x

    d_prev = None
    r_prev = None
    rs = []
    xs = [x.copy()]

    num_iter = 0
    while True:
        ri = residual(x)
        r_norm = np.linalg.norm(ri)
        rs.append(r_norm)
        if np.linalg.norm(ri) <= 1e-6:
            break
        riri = ri @ ri
        if num_iter == 0:
            di = ri
        else:
            di = ri + riri / (r_prev @ r_prev) * d_prev
        alphai = riri / (di @ A @ di)
        x += alphai * di

        xs.append(x.copy())
        num_iter += 1
        r_prev = ri
        d_prev = di
    return x, np.stack(xs), rs

n = 2000
A = generate_random_psd_matrix(n, definite=True)
x_star = np.random.randn(n)
b = A @ x_star

tic = time.time()
x_cg, xs, rs = conjugate_gradient_method(A,b)
toc = time.time()

es = np.linalg.norm(xs - x_star, axis=1)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(rs)
ax1.set_title('Residual norm')

ax2.plot(es)
ax2.set_title('Error norm')

plt.legend()
plt.show()

print(toc - tic)

tic = time.time()
x_np = np.linalg.solve(A,b)
toc = time.time()
print(toc-tic)

print(np.linalg.norm(x_np-x_cg))
