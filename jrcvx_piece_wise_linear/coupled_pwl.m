load('data_coupled.mat');
m = size(A1, 1);
n = size(A1, 2);
b1 = b1';
b2 = b2';

cvx_begin
		variables x1(n-1) x2(n-1) y
		minimize (max(A1*[x1; y] + b1) + max(A2*[x2; y] + b2))
cvx_end


