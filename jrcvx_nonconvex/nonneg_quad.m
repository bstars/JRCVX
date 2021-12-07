load('nonneg_quad.mat');
q = q';
n = size(q,1)

t = 1;
r = 1;



% cvx_begin
% 	variable x(n,1)
% 	minimize (t * (0.5 * x' * P * x + q' * x + r) - sum(log(x)))
% cvx_end

cvx_begin
	variable x(n,1)
	dual variable lamb
	minimize (0.5 * x' * P * x + q' * x + r)
	subject to
		lamb : x >= 0;
cvx_end


