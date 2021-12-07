load('lp.mat');
b = b';
c = c';

m = size(A,1);
n = size(A,2);

cvx_begin
	variable x(n,1)
	dual variable lamb
	minimize (c' * x)
	subject to
		lamb : x >= 0;
		A * x == b;
cvx_end