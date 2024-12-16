function [x, ks, output] = gl_ProxGD_primal(x0, A, b, mu, opts_out)
    % Solve the group LASSO problem using the Proximal g Method.
    opts_out = ProxGD_primal_optsInit(opts_out);
    opts_out.method = @gl_ProxGD_primal_inner;
    [x, ks, output] = LASSO_group_con(x0, A, b, mu, opts_out);
end

