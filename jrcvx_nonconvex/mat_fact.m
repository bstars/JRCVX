load('mat_fact.mat');

m = 50; n = 50; k = 5;
% A = rand(m,k)*rand(k,n);

% Initialize Y randomly
Y = rand(m,k);

% Perform alternating minimization
MAX_ITERS = 30;
residual = zeros(1,MAX_ITERS);
for iter = 1:MAX_ITERS
	cvx_begin
		cvx_quiet(true);
		if mod(iter,2) == 1
			variable X(k,n)
		else
			variable Y(m,k)
		end
		X >= 0; 
		Y >= 0;
		minimize(norm(A - Y*X,'fro'));
	cvx_end
	fprintf(1,'Iteration %d, residual norm %g\n',iter,cvx_optval);
	residual(iter) = cvx_optval;
end

% Display results
disp( 'Original matrix:' );
disp( A );
disp( 'Left factor Y:' );
disp( Y );
disp( 'Right factor X:' );
disp( X );
disp( 'Residual A - Y * X:' );
disp( A - Y * X );
fprintf( 'Residual after %d iterations: %g\n', iter, cvx_optval );

% Plot residuals
plot(residual); 
hold on;

