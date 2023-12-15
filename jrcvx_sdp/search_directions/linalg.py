import numpy as np
import scipy.sparse as SP
import scipy.sparse.linalg as SPLA

def operator_A(As, X):
	"""
	:param As: np.array, of shape [k,n,n]
	:param X:  np.array, of shape [n,n]
	:return: A(X)
	"""

	return np.sum(As * X, axis=(1,2))

def operator_AT(As, v):
	"""
	:param As: np.array, of shape [k,n,n]
	:param v:  np.array, of shape [k]
	:return: A^T(v)
	"""
	return np.sum(np.transpose(As, (1,2,0)) * v, axis=-1)

def mat_index_to_vec_index_multi(n, idx):
	"""
	:param n:
	:param idx: Of shape [k,2]
	:return:
	"""
	return idx[:,1] * n + idx[:,0]

def mat_index_to_svec_index_multi(n, idx):
	"""
	Convert the index of a element in a matrix U to its index in svec(U)
	:param n:
	:param idx: Of shape [k,2], each row is a index of a lower-triangular element
	:return:
	"""
	return ((n - idx[:,1]) + n-1) * idx[:,1] // 2 + idx[:,0]

def generate_Q(n):
	"""
	Generate the matrix Q \in R^{(1+n) * n / 2, n*n} such that for a symmetric matrix U \in S^{n}
			svec(U) = Q vec(U)
	and
			vec(U) = Q^T svec(U)
	:return: Q, of type scipy.sparse.csr_matrix
	"""
	row_index = []
	col_index = []
	elems = []

	# deal with the diagonal elements of U
	r_index = np.arange(n, 1, -1)
	r_index = np.cumsum(r_index)
	r_index = np.concatenate([[0], r_index])
	index = np.arange(0,n)
	index = np.stack([index, index], axis=1)
	c_index = mat_index_to_vec_index_multi(n, index)

	row_index.extend(r_index)
	col_index.extend(c_index)
	elems.extend(np.ones([n]))

	# deal with the off-diagonal elements of U
	num_lower_tri = (n * (n-1))//2  # number of off-diagonal lower-triangular elements
	index = np.tril_indices(n,k=-1)
	index = np.stack([index[0], index[1]]).T
	r_index = mat_index_to_svec_index_multi(n, index)
	r_index = np.concatenate([r_index, r_index])

	index = np.concatenate([index, index[:, ::-1]])
	c_index = mat_index_to_vec_index_multi(n, index)


	row_index.extend(r_index)
	col_index.extend(c_index)
	elems.extend(np.ones([num_lower_tri * 2]) * 0.5 * np.sqrt(2))
	# elems.extend(np.ones([num_lower_tri * 2]) * 0.5)


	Q = SP.csr_matrix((elems, (row_index, col_index)), shape=[(n * (n+1))//2, n*n])
	return Q

def vec(U):
	"""
	Stack all columns of U
	"""
	return np.reshape(U.T, newshape=[-1])

def vec_inv(v):
	"""
	Compute the matrix U such that vec(U) = v
	"""
	n = int(np.sqrt(len(v)))
	return np.reshape(v, [n,n]).T

def svec(U, matmaul=True):
	"""
	For any symmetric U,
	compute the svec(U) = [U[1,1], sqrt(2)U[2,1], ..., sqrt(2)U[n,1], U[2,2], sqrt(2)U[3,2], ... ]
	"""
	n,n = U.shape
	if not matmaul:
		__vec = U[np.triu_indices(n)]
		idx = np.tril_indices(n,k=-1)
		idx = np.stack([idx[0], idx[1]]).T
		svec_idx = mat_index_to_svec_index_multi(n, idx)
		mask = np.ones_like(__vec)
		mask[svec_idx] = np.sqrt(2)
		__vec *= mask
		return __vec
	else:
		Q = generate_Q(n)
		return Q.dot(vec(U))

def svec_inv(v):
	n = int(np.sqrt(2 * len(v)))
	Q = generate_Q(n)
	return vec_inv(Q.transpose().dot(v))

def svec_multi(As):
	ret = []
	for A in As:
		ret.append(svec(A))
	return np.stack(ret).T

def symkro(A,B):
	"""
	Compute the symmetric kronecker product of two square matrix
	:param A: Of shape [n,n]
	:param B: Of shape [n,n]
	:return:
	"""
	assert A.shape[0] == A.shape[1]
	assert B.shape[0] == B.shape[1]
	assert A.shape[0] == B.shape[0]

	n = A.shape[0]

	AB = SP.kron(A,B)
	BA = SP.kron(B,A)
	Q = generate_Q(n)
	return 0.5 * Q.dot(AB + BA).dot(Q.transpose()).toarray()



if __name__ == '__main__':

	pass


