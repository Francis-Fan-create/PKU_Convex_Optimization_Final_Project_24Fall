function [x, num_iters, out] = gl_ADMM_primal(initial_x, A, b, mu, opts)
    % Solve the primal problem of the group LASSO using the ADMM algorithm.
    %
    % Args:
    %   initial_x (matrix): Initial guess for the variable x
    %   A (matrix): Constraint matrix
    %   b (matrix): Observation vector
    %   mu (double): Regularization parameter
    %   opts (struct): Algorithm options (optional)
    %
    % Returns:
    %   x (matrix): Updated x
    %   num_iters (int): Number of iterations
    %   out (struct): Output structure with algorithm details
    
    if nargin < 5
        opts = struct();
    end
    
    % Initialize options with default ADMM primal parameters
    opts = utils.ADMM_primal_optsInit(opts);
    [~, num_features] = size(A);
    [~, num_targets] = size(b);
    out = utils.outInit();
    out.prim_hist = [];
    
    x = initial_x;
    y = initial_x;
    z = zeros(num_features, num_targets);
    
    sigma = opts.sigma;   % Quadratic penalty coefficient
    % Since the penalty factor doesn't change during iterations,
    % precomputing the inverse can accelerate the process
    inverse_matrix = inv(sigma * eye(num_features) + A' * A);
    A_transpose_b = A' * b;
    
    for iteration = 1:opts.maxit
        % Update x
        x = inverse_matrix * (sigma * y + A_transpose_b - z);
        
        % Store previous y for convergence check
        previous_y = y;
        
        % Update y using the proximal operator
        y = utils.prox(x + z / sigma, mu / sigma);
        
        % Update dual variable z
        z = z + sigma * (x - y);
        
        % Calculate primal and dual residuals for convergence
        primal_residual = norm(x - y, 'fro');
        dual_residual = norm(previous_y - y, 'fro');
        
        % Compute the primal objective function value
        objective_value = utils.objFun(x, A, b, mu);
        out.prim_hist = [out.prim_hist; objective_value];
        
        % Check for convergence
        if primal_residual < opts.thre && dual_residual < opts.thre
            break;
        end
    end
    
    % Store the number of iterations and final objective value
    out.itr = iteration;
    out.fval = objective_value;
    fprintf('ADMM_primal: itr: %d, fval: %f\n', out.itr, out.fval);
    fprintf('ADMM_primal: len(out.prim_hist): %d\n', length(out.prim_hist));
    
    % Prepare iteration history for output
    iters_range = 1:out.itr;
    out.iters = [iters_range', out.prim_hist];
    
    num_iters = out.itr;
end