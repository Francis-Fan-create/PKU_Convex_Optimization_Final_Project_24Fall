function [x, num_iters, out] = gl_ADMM_dual(initial_x, A, b, mu, opts)
    % Solve the dual problem of the group LASSO using the ADMM algorithm
    %
    % Args:
    %   initial_x (matrix): Initial guess for the primal variable x
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
    
    % Initialize options with default ADMM dual parameters
    opts = utils.ADMM_dual_optsInit(opts);
    fprintf('optsOuter:\n'); disp(opts);
    
    [m, n] = size(A);
    [~, l] = size(b);
    out = utils.outInit();
    out.prim_hist = [];
    out.dual_hist = [];
    
    x = initial_x;
    sigma = opts.sigma;   % Quadratic penalty coefficient
    inverse_matrix = inv(eye(m) + sigma * A * A');
    z = zeros(n, l);
    
    for iteration = 1:opts.maxit
        % Update dual variable y
        y = inverse_matrix * (A * x - sigma * A * z - b);
        
        % Store previous z for convergence check
        previous_z = z;
        
        % Update primal variable z using the proximal operator
        z = updateZ(x / sigma - A' * y, mu);
        
        % Calculate the change in z for convergence
        z_change = norm(z - previous_z, 'fro');
        
        % Update primal variable x
        x = x - sigma * (A' * y + z);
        
        % Compute the primal objective function value
        primal_obj = utils.objFun(x, A, b, mu);
        
        % Compute the dual objective function value
        dual_obj = 0.5 * norm(y, 'fro')^2 + sum(sum(y .* b));
        
        % Record history of objective values
        out.prim_hist = [out.prim_hist; primal_obj];
        out.dual_hist = [out.dual_hist; dual_obj];
        
        % Check convergence based on the norm of the residual
        if norm(A' * y + z, 'fro') < opts.thre
            break;
        end
    end
end

function [z] = updateZ(reference, mu)
    % Update the variable Z using the proximal operator
    %
    % Args:
    %   reference (matrix): Reference matrix for updating Z
    %   mu (double): Regularization parameter
    %
    % Returns:
    %   matrix: Updated Z matrix
    
    norm_ref = vecnorm(reference, 2, 2);  % Row-wise L2 norm
    norm_ref = reshape(norm_ref, [], 1);   % Ensure column vector
    mask = norm_ref < mu;
    norm_ref(mask) = mu;
    updated_z = reference .* (mu ./ norm_ref);
    z = updated_z;
end

