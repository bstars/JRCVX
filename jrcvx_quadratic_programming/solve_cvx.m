clear all;
load('data.mat')
q = q';
n = size(q, 1);

cvx_begin
	variable x(n,1)
	minimize (0.5 * x' * P * x + q' * x)
	subject to
		x <= 1;
		x >= -1;
cvx_end