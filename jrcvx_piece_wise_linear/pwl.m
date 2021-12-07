load('data.mat')
b = b';

m = size(A,1);
n = size(A,2);

% cvx_begin
% 	variable x(n-1)
% 	variable y
% 	minimize (max(A*[x; y] + b))
% 	subject to
% 		y == 1;
% cvx_end

cvx_begin
	variable x(n-1)
	variable y
	minimize (max(A*[x; y] + b))
	subject to
		y == 1;
cvx_end

