function [x, iter, out] = gl_FProxGD_primal(x0, A, b, mu, opts_out)
    % Fast Proximal g Method for Group LASSO Problem
    %
    % Args:
    %     x0 (matrix): Initial guess for the variable x.
    %     A (matrix): Constraint matrix.
    %     b (matrix): Observation vector.
    %     mu (float): Regularization parameter.
    %     opts (struct, optional): Algorithm options.
    %
    % Returns:
    %     x (matrix): Solution x.
    %     iter (int): Number of ks.
    %     out (struct): Output information.

    opts_out = FProxGD_primal_optsInit(opts_out);
    opts_out.method = @gl_FProxGD_primal_inner;
    [x, iter, out] = LASSO_group_con(x0, A, b, mu, opts_out);
end

