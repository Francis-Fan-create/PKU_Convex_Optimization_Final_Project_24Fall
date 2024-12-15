function [x, iter, output] = gl_ADMM_primal(x0, A, b, mu, opts)
    % Solve the primal problem of the group LASSO using the ADMM algorithm.
    %
    % Args:
    %     x0 (matrix): Initial guess for the variable x.
    %     A (matrix): Constraint matrix.
    %     b (matrix): Observation vector.
    %     mu (float): Regularization parameter.
    %     opts (struct, optional): Algorithm options.
    %
    % Returns:
    %     x (matrix): Updated x.
    %     iter (int): Number of ks.
    %     output (struct): Output information.

    % Initialize options with default ADMM primal parameters
    opts = ADMM_primal_optsInit(opts);

    [~, n] = size(A);
    [~, l] = size(b);
    output = outInit();
    output.prim_hist = [];

    x = x0;
    y = x0;
    z = zeros(n, l);

    sigma = opts.sigma;   % Quadratic penalty coefficient
    % Since the penalty factor does not change during ks, precomputing the inverse can accelerate the process.
    A_transpose_b = A' * b;

    for k = 1:opts.maxit
        % Update x
        x = (sigma * eye(n) + A' * A)\(sigma * y + A_transpose_b - z);
        
        % Store previous y for convergence check
        previous_y = y;
        
        % Update y using the proximal operator
        y = prox(x + z / sigma, mu / sigma);
        
        % Update dual variable z
        z = z + sigma * (x - y);

        % Calculate primal and dual residuals for convergence
        primal_residual = norm(x - y);
        dual_residual = norm(previous_y - y) ;
        
        % Compute the primal objective function value
        objective_value = objFun(x, A, b, mu);
        output.prim_hist = [output.prim_hist; objective_value];

        % Check for convergence
        if primal_residual < opts.thre && dual_residual < opts.thre
            break;
        end
    end

    % Store the number of ks and final objective value
    output.iter = k;
    output.fval = objective_value;
    
    % Prepare k history for output
    output.iters = [ (1:output.iter)', output.prim_hist ];
    
    iter = output.iter;
end

