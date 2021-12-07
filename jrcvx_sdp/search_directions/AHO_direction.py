import numpy as np
import scipy.sparse as SP
import scipy.sparse.linalg as SPLA

import sys
sys.path.append('../../..')

import JRCVX.jrcvx_sdp.search_directions.linalg as jrlinalg
from JRCVX.utils.random_generate import generate_random_psd_matrix


def residual(As, C, b, X, Lamb, v, t):
	n,n = X.shape
	rp = jrlinalg.operator_A(As, X) - b # primal residual
	rd = jrlinalg.operator_AT(As, v) + Lamb - C # dual residual
	rc = - 1/t * SP.eye(n) + 0.5 * (X @ Lamb + Lamb @ X) # centrality residual
	return rp, rd, np.array(rc)

def residual_norm(As, C, b, X, Lamb, v, t):
	rp, rd, rc = residual(As, C, b, X, Lamb, v, t)
	return np.sqrt(
		np.sum(np.square(rp)) + np.sum(np.square(rd)) + np.sum(np.square(rc))
	)


def solve_AHO_system(As, C, b, X, Lamb, v, t, svec_As=None, alpha=0.01, beta=0.5, sym_tol=1e-8):
	"""
	Solve for the linear system for AHO direction

		[0           SVEC(As)^T      0         ]  [ dv         ]      [b - A(x)                             ]
		[SVEC(As)    0               I         ]  [ svec_dX    ]  =   [svec(C - A^T(v) - Lamb)              ]
		[0           I \skro Lamb    I \skro X ]  [ svec_dLamb ]      [svec(1/t I - 0.5*(X*Lamb + Lamb*X))  ]
	with block elimination
	:param As: np.array, of shape [k, n, n]
	:param b:  np.array, of shape [k,]
	:param v:  np.array, the dual variable associated with equality constraint A(X) = b, of shape [k,]
	:param svec_X: np.array, svec of current value of primal variable X
	:param svec_Lamb: np.array, svec of current value of dual variable Lamb associated with inequality constraint X >= 0
	:param alpha: Line search parameter
	:param beta: Line search parameter
	:return:
		dX: Search direction for primal variable X
		dlamb: Search direction for dual variable Lamb
		dv: Search direction for dual variable v
		s: Step size (achieved by line search on the residual norm)

	"""
	assert np.all(np.abs(C - C.T) < sym_tol)
	assert np.all(np.abs(X - X.T) < sym_tol)
	assert np.all(np.abs(Lamb - Lamb.T) < sym_tol)
	assert np.all(np.abs(As - np.transpose(As, [0,2,1])) < sym_tol)

	n,n = X.shape
	svec_As = jrlinalg.svec_multi(As) if svec_As is None else svec_As

	rp, rd, rc = residual(As, C, b, X, Lamb, v, t)
	svec_rd = jrlinalg.svec(rd)
	svec_rc = jrlinalg.svec(rc)

	I_skro_Lamb = jrlinalg.symkro(SP.eye(n), Lamb)
	I_skro_X = jrlinalg.symkro(SP.eye(n), X)

	I_skro_Lamb_inv = np.linalg.inv(I_skro_Lamb)    # This step is too expansive
	# I_skro_Lamb_inv = np.eye(I_skro_Lamb.shape[0])
	dv_rhs = svec_As.T @ I_skro_Lamb_inv @ (-svec_rc + I_skro_X @ svec_rd) + rp
	dv_lhs = -svec_As.T @ I_skro_Lamb_inv @ I_skro_X @ svec_As
	dv = np.linalg.solve(dv_lhs, dv_rhs)    # solve for dv according to row 1

	svec_dLamb = -svec_rd - svec_As @ dv       # row 2
	svec_dX = I_skro_Lamb_inv @ ( -svec_rc - I_skro_X @ svec_dLamb) # row 3

	# print(np.linalg.norm(svec_As.T @ svec_dX + rp))
	# print(np.linalg.norm(svec_As @ dv + svec_dLamb + svec_rd))
	# print(np.linalg.norm(I_skro_Lamb @ svec_dX + I_skro_X @ svec_dLamb + svec_rc))

	dX = jrlinalg.svec_inv(svec_dX)
	dLamb = jrlinalg.svec_inv(svec_dLamb)

	# print(np.linalg.norm(jrlinalg.operator_A(As, dX) + rp))
	# print(np.linalg.norm(jrlinalg.operator_AT(As, dv) + dLamb + rd))
	# print(np.linalg.norm(
	# 	0.5 * (dX @ Lamb + Lamb @ dX) + 0.5 * (dLamb @ X + X @ dLamb) + rc
	# ))
	s = 1   # step size

	evalues, _ = np.linalg.eig(X + s * dX)
	while np.min(evalues) <= 0:
		s *= beta
		evalues, _ = np.linalg.eig(X + s * dX)

	evalues, _ = np.linalg.eig(Lamb + s * dLamb)
	while np.min(evalues) <= 0:
		s *= beta
		evalues, _ = np.linalg.eig(Lamb + s * dLamb)

	norm = residual_norm(As, C, b, X, Lamb, v, t)
	while (1 - alpha * s) * norm <= residual_norm(As, C, b, X+s*dX, Lamb+s*dLamb, v+s*dv, t):
		s *= beta

	return dX, dLamb, dv, s



if __name__ == "__main__":
	k = 2
	n = 10
	As = []
	for i in range(k):
		As.append(
			jrlinalg.generate_sym(n)
		)
	As = np.stack(As)

	C = jrlinalg.generate_sym(n)
	b = np.random.randn(k)

	X = generate_random_psd_matrix(n, definite=True)
	Lamb = generate_random_psd_matrix(n, definite=True)
	v = np.random.randn(k)


	solve_AHO_system(As, C, b, X, Lamb, v, 1)

