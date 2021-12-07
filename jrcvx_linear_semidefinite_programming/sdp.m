load('data.mat');
b = b';
c = c';

[p,p,n] = size(F);

cvx_begin sdp
    variable x(n,1)
    dual variable lamb
    dual variable v
    minimize (c' * x)
    subject to
        v : A * x == b;
        lamb : x(1) * F(:,:,1) + \
        x(2) * F(:,:,2) + \
        x(3) * F(:,:,3) + \
        x(4) * F(:,:,4) + \
        x(5) * F(:,:,5) + \
        x(6) * F(:,:,6) + \
        x(7) * F(:,:,7) + \
        x(8) * F(:,:,8) + \
        x(9) * F(:,:,9) + \
        x(10) * F(:,:,10) + \
        x(11) * F(:,:,11) + \
        x(12) * F(:,:,12) + \
        x(13) * F(:,:,13) + \
        x(14) * F(:,:,14) + \
        x(15) * F(:,:,15) + \
        x(16) * F(:,:,16) + \
        x(17) * F(:,:,17) + \
        x(18) * F(:,:,18) + \
        x(19) * F(:,:,19) + \
        x(20) * F(:,:,20) + G <= 0;
cvx_end







