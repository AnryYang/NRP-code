function [w] = update_dw(min_w, lambda, h, D1, D2, w1, w2, X1, X2, X1sq, X2sq, XY, XYsq)
    w2sq = w2.^2;
    Xi_2 = sum(bsxfun(@times,X2, (D2.*w2)'));
    Chi_2 = sum(bsxfun(@times,X2, w2'));
    Phi_2 = sum(bsxfun(@times,X2sq, w2sq'));
    X1_w1 = bsxfun(@times,X1, w1');
    Rho_1 = sum(X1_w1);
    Rho_2 = sum(bsxfun(@times,X2, (w2sq.*w1.*XY)' ));
    X2_w2sq = bsxfun(@times,X2, w2sq');

    Lambda_2 = X2_w2sq'*X2;

    W_XY = XY.*w2;
    Chi_x_c = (X1*Chi_2')'-W_XY;
    a = (X1*Xi_2')' + D1.*Chi_x_c;
    b = h.*(sum(bsxfun(@times, X1sq', Phi_2'))- XYsq.*w2sq ) + Chi_x_c.^2 + lambda;
    Lambda_2_X1 = X1*Lambda_2';
    Rho_2_X1 =  w1.*XYsq.*w2sq-Rho_2*X1';

    [row, col] = find(b>0);
    col_size = numel(col);
    for j=1:col_size
        i = col(j);
        a3 = (Rho_1-X1_w1(i,:))*Lambda_2_X1(i,:)'+ Rho_2_X1(i);
        w_i = w1(i);
        w1(i) = (a(i)-a3)/b(i);
        w1(i) = max(min_w, w1(i));
        Rho_1 = Rho_1 + (w1(i)-w_i).*X1(i,:);
        Rho_2 = Rho_2 + ((w1(i)-w_i)*XY(i)).*X2_w2sq(i,:);
    end

    w = w1;
end
