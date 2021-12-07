load('noncvx_qp.mat')
q = q';
Kmax = 30; Nmax = 10;
n = size(q,1);

for i = 1:Nmax
	% Choose a random starting point.
    xk = 2*(rand(n, 1) - 0.5);

    fs = zeros(Kmax, 1);
    for k = 1:Kmax
        fxk = (1/2)*xk'*P*xk + q'*xk;
        cvx_begin
            [V, D] = eig(P);
            Pp = V*pos(D)*V';

            variable x(n);

            minimize(fxk + (P*xk + q)'*(x - xk) + (1/2)*quad_form(x - xk, Pp))
            norm(x, inf) <= 1;
			abs(x - xk) <= 0.2;
        cvx_end
        disp(cvx_status); disp(cvx_optval);
        fs(k) = cvx_optval;

		% Stop if we have changed less than 0.1%.
        if norm(xk - x) <= 0.001*norm(x)
            fs(k+1:end) = cvx_optval;
            break;
        end
        xk = x;
    end
    % plot(fs); hold on; drawnow;
end

if 0 % produce graphs.
	axis([1 Kmax -70 -10])
	% plot([1 Kmax], [lowerbound lowerbound], 'k--')
	xlabel('x')
	ylabel('y')
	hold on;
	drawnow;
	% print -deps ncqp.eps
end