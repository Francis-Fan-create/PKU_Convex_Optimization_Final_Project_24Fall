function [x, iter, out] = gl_cvx_mosek(x0, A, b, mu, opts)
    % Solve the group LASSO problem using CVX with MOSEK solver.
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
    %     iter (int): Number of ks (not provided by CVX).
    %     out (struct): Output information.

    [m, n] = size(A);
    [~, l] = size(b);
    % CVX modeling with MOSEK solver
    cvx_begin
        cvx_solver mosek
        variable x(n, l)
        % Objective function: minimize 0.5 * ||A * x - b||_F^2 + mu * sum of L2 norms of rows of x
        minimize( 0.5 * sum(sum_square(A * x - b, 2)) + mu * sum(norms(x, 2, 2)) )
    cvx_end

    % Since CVX does not provide k count, set iter to empty
    iter = -1;

    % Prepare output structure
    out = struct();
    out.iter = iter;  % Number of ks
    out.fval = cvx_optval;  % Final objective function value

end