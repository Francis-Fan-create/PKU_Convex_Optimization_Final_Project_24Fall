function [x, iter, output] = gl_ADMM_dual(x0, A, b, mu, opts)
    % Solve the dual problem of the group LASSO using the ADMM algorithm.
    %
    % Args:
    %     x0 (matrix): Initial guess for the primal variable x.
    %     A (matrix): Constraint matrix.
    %     b (matrix): Observation vector.
    %     mu (float): Regularization parameter.
    %     opts (struct, optional): Algorithm options.
    %
    % Returns:
    %     x (matrix): Updated x.
    %     iter (int): Number of ks.
    %     output (struct): Output information.

    % Initialize options with default ADMM dual parameters
    opts = ADMM_dual_optsInit(opts);

    [m, n] = size(A);
    [~, l] = size(b);
    output = outInit();
    output.prim_hist = [];
    output.dual_hist = [];

    x = x0;
    sigma = opts.sigma;   % Quadratic penalty coefficient
    z = zeros(n, l);

    for k = 1:opts.maxit
        % Update dual variable y
        y = (eye(m) + sigma * (A * A'))\(A * x - sigma * A * z - b);

        % Update primal variable z using the proximal operator
        z = updateZ(x ./ sigma - A' * y, mu);

        % Update primal variable x
        x = x - sigma * (A' * y + z);

        % Compute the primal objective function value
        primal_obj = objFun(x, A, b, mu);

        % Compute the dual objective function value
        dual_obj = 0.5 * norm(y, 'fro') ^ 2 + sum(y .* b);

        % Record history of objective values
        output.prim_hist = [output.prim_hist; primal_obj];
        output.dual_hist = [output.dual_hist; dual_obj];

        % Check convergence based on the norm of the r
        if norm(A' * y + z) < opts.thre
            break;
        end
    end

    % Store the number of ks and final objective value
    output.iter = k;
    output.fval = output.prim_hist(end);

    % Prepare k history for output
    output.iters = [ (1:output.iter)', output.prim_hist];

    iter = output.iter;
end

function updated_z = updateZ(reference, mu)
    % Update the variable Z using the proximal operator.

    % Compute the norms of each row
    norm_ref = sqrt(sum(reference .^ 2, 2));

    % Adjust norms less than mu to mu
    norm_ref_adj = norm_ref;
    norm_ref_adj(norm_ref_adj < mu) = mu;

    % Compute updated_z
    updated_z = reference .* (mu ./ norm_ref_adj);
end

