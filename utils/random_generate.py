import numpy as np

def generate_random_symmetric_matrix(n):
    A = np.random.randn(n, n)
    A = A.T @ A
    return A

def generate_random_psd_matrix(n, definite=False):
    X = generate_random_symmetric_matrix(n)
    evals, evecs = np.linalg.eig(X)
    eps = 1 if definite else 0
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(evals) @ evecs.T

def generate_random(*dn):
    return np.random.randn(*dn)

