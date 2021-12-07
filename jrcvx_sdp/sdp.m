load('data.mat');
n = size(As,2)

cvx_begin % sdp
    variable X(n,n) symmetric
    dual variable Lamb
    dual variable v1
    dual variable v2
    dual variable v3
    minimize trace(C*X)
    subject to
            v1: trace(As(:,:,1) * X) == b(1);
            v2: trace(As(:,:,2) * X) == b(2);
            % v3: trace(As(:,:,3) * X) == b(3);
            Lamb: X == semidefinite(n);
cvx_end

%A1=[1 0 1; 0 3 7; 1 7 5];
%A2=[0 2 8; 2 6 0; 8 0 4];
%C=[1 2 3; 2 9 0; 3 0 7];
%n=size(C,1);
%cvx_begin
% variable X(n,n) symmetric;
% minimize trace(C*X)
% subject to
% trace(A1*X) == 11;
% trace(A2*X) == 19; % alt: sum(sum(A2.*X)) == 19
% X == semidefinite(n);
%cvx_end