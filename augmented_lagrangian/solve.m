load('data.mat');

m = size(A,1);
n = size(A,2);
b = b';

cvx_begin
	variable x(n)
	minimize norm(x)
	subject to
		A * x == b;
cvx_end
