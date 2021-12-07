

import numpy as np
import matplotlib.pyplot as plt


def flow_control_dual_decomposition_method(R:np.array, c:np.array, eps=1e-3):
    """
    Solve the flow-control problem with log flow utility function
        min(f). \sum \log f
        s.t.    Rf <= c
    where R gives topology of the graph and c is the capacity constraint
    """
    m,n = R.shape

    lamb = np.ones(shape=[m]) * 2
    primal_objs = []
    dual_objs = []

    k = 1
    while True:
        k += 1
        # solve the subproblem
        #   min(fj). lamb @ rj * fj - log fj
        f = 1. / (R.T @ lamb)


        # dual function value
        # g(lamb) = - lamb @ c + inf(f)(lamb @ R @ f - \sum \log f)
        dual_obj = -1 * np.inner(lamb, c) + lamb @ R @ f - np.sum(np.log(f))
        dual_objs.append(dual_obj)



        # construct feasible flows
        f_feas = f.copy()
        eta = (R @ f) / c
        for j in range(n):
            link_on_flow = np.where(R[:,j]>0)[0]
            f_feas[j] = f[j] / np.max(eta[link_on_flow]) # - 1e-9



        # '''
        primal_obj = -1 * np.sum(np.log(f_feas))
        primal_objs.append(primal_obj)

        # print(dual_obj - dual_objs[-2], primal_objs[-2] - primal_obj)



        if np.abs(dual_obj - primal_obj) <= eps:
            return f_feas, np.array(primal_objs), np.array(dual_objs)

        step_size = 1
        g = c - R @ f
        lamb = np.maximum(0, lamb - step_size * g)
        # '''




if __name__ == '__main__':

    R = np.array([
        [0,0,1,0,0,1,0,0,0,0,0,1],
        [1,0,0,1,0,0,0,1,1,0,0,0],
        [0,0,1,0,0,0,0,0,1,0,1,0],
        [0,0,0,0,1,0,1,0,0,0,0,1],
        [1,1,0,0,0,0,1,0,0,0,1,0],
        [0,0,0,0,0,1,0,1,0,1,0,0],
        [0,0,1,0,0,0,0,1,0,1,0,0],
        [1,1,0,1,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,1,0,0,1,1,0,0],
        [1,0,0,0,1,0,0,1,0,0,1,0]
    ], dtype=np.float)

    m,n = R.shape

    c = np.random.rand(m) * 0.9 + 0.1


    f, primal_objs, dual_objs = flow_control_dual_decomposition_method(R, c)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(primal_objs, label='primal values')
    ax1.plot(dual_objs, label='dual values')
    ax1.legend()

    ax2.plot(primal_objs - dual_objs, label='duality gap')
    ax2.legend()
    plt.show()