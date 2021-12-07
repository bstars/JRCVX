import matplotlib.pyplot as plt
import numpy as np

def vals(x,y,coordx,coordy,P):
    return np.array([x-coordx, y-coordy]) @ P @ np.array([x-coordx, y-coordy])

def plot_2d_ellipsoid(u, P, levels = [1]):
    ux = u[0]
    uy = u[1]
    x = np.linspace(-4, 8, 100)
    y = np.linspace(-6, 8, 100)
    # x = np.linspace(-3,3,100)
    # y = np.linspace(-3, 3, 100)
    xs, ys = np.meshgrid(x, y)
    val = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            val[i, j] = vals(xs[i, j], ys[i, j], ux, uy,P)
    plt.contour(xs, ys, val, levels)
    plt.plot(ux, uy, 'ro')

def show():
    plt.tight_layout()
    plt.gca().set_aspect("equal")
    plt.grid()
    plt.show()

def plot_path(xs):
    plt.plot(xs[:,0], xs[:,1],'b--')

