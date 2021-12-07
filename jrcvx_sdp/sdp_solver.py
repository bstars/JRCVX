import sys
sys.path.append('../..')

import numpy as np


import JRCVX.jrcvx_sdp.search_directions.AHO_direction as aho
import JRCVX.jrcvx_sdp.search_directions.linalg as jrlinalg
from JRCVX.utils.random_generate import generate_random_psd_matrix

def optimality_conditions(C, As, b, X, Lamb, v):
	rp = jrlinalg.operator_A(As, X) - b
	rd = jrlinalg.operator_AT(As,v) + Lamb - C
	gap = np.sum(X * Lamb)
	return rp, rd, gap


def sdp_solve(C, As, b, direction='AHO', mu=3, eps=1e-6):
	"""
	Solve the semi-definite programming
		min.    <C,X>
		s.t.    <Ai,X> = bi
				X is PSD
	with infeasible primal-dual path-following method

	:param direction:
	:return:
	"""
	assert C.shape[0] == C.shape[1]

	print('Calling Infeasible Primal-Dual Path-Following Method with %s direction' % (direction))

	svec_As = jrlinalg.svec_multi(As)

	k,n,n = As.shape
	X = np.eye(n)
	Lamb = np.eye(n)
	v = np.ones(shape=[k]) * 0.1

	num_iter = 0
	while True:

		rp, rd, gap = optimality_conditions(C, As, b, X, Lamb, v)
		nrp = np.linalg.norm(rp)
		nrd = np.linalg.norm(rd)

		print("Iteration: %d, primal residual: %.5f, dual residual %.5f, surrogate duality gap: %.5f"
		      % (num_iter, nrp, nrd, gap))
		if nrp <= eps and nrd <= eps and gap <= eps:
			obj = np.sum(C * X)
			dual_obj = np.sum(b * v)
			print("Num of iterations        = %d" % (num_iter))
			print("Primal objective         = %.8f" % (obj))
			print("Dual objective           = %.8f" % (dual_obj))
			print("Duality gap trace(XZ):   = %.8f" % (gap))
			return X, Lamb, v


		t = n / gap * mu

		dX, dLamb, dv, s = aho.solve_AHO_system(As, C, b, X, Lamb, v, t, svec_As=svec_As)

		X += s * dX
		Lamb += s * dLamb
		v += s * dv
		num_iter += 1



if __name__ == '__main__':
	k = 2
	n = 2
	As = []
	for i in range(k):
		As.append(
			generate_random_psd_matrix(n)
		)
	As = np.stack(As)
	X = generate_random_psd_matrix(n, definite=True)
	b = jrlinalg.operator_A(As, X)
	C = generate_random_psd_matrix(n)


	# A1 = [
	# 	[1.,0,1],[0,3,7],[1,7,5]
	# ]
	# A2 = [
	# 	[0., 2, 8], [2, 6, 0], [8,0,4]
	# ]
	#
	# C = [
	# 	[1.,2,3],[2,9,0],[3,0,7]
	# ]
	#
	# b = [11,19.]
	#
	# As = np.array([A1, A2])
	#
	# b = np.array(b)
	# C = np.array(C)


	from scipy.io import savemat
	savemat(
		'data.mat',
		mdict={
			'As' : np.transpose(As,[1,2,0]),
			'b' : b,
			'C' : C
		}

	)

	X, Lamb, v = sdp_solve(C, As, b,)
	# print(X)
	# print(Lamb)
	# print(v)



