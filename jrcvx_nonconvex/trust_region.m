clear all;
load('trust_region.mat')
q = q';
n = size(q, 1);
xk = xk';

t = 1;
r = 1;
rou = 0.01;

a = xk - rou;
b = xk + rou;

% cvx_begin
% 	variable x(n,1)
% 	minimize (t*(0.5 * x' * P * x + q'*x + r) - sum(log(x+1)) - sum(log(1-x)) - sum(log(x-a)) - sum(log(b-x)))
% cvx_end

cvx_begin
	variable x(n,1)
	minimize (0.5 * x'*P*x + q'*x + r)
	subject to
		norm(x, inf) <= 1;
		norm(x-xk, inf) <= rou;
cvx_end