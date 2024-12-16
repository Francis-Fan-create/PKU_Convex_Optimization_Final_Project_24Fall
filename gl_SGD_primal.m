function [x, ks, output] = gl_SGD_primal(x0, A, b, mu, opts_out)
    % sub_g method for the primal problem.
    % - Minimize (1/2)*||Ax - b||_2^2 + mu*||x||_{1,2}
    % - A is m x n, b is m x l, x is n x l
    opts_out = SGD_primal_optsInit(opts_out);
    opts_out.method = @gl_SGD_primal_inner;
    [x, ks, output] = LASSO_group_con(x0, A, b, mu, opts_out);
end

