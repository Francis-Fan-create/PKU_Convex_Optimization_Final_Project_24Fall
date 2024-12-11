function [x, num_iters, out] = gl_ALM_dual(initial_x, A, b, mu, opts)
    % Solve the dual problem of the group LASSO using the Augmented Lagrangian Method (ALM).
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
    
    % Initialize options with default ALM dual parameters
    opts = utils.ALM_dual_optsInit(opts);
    fprintf('optsOuter:\n'); disp(opts);
    
    [m, n] = size(A);
    [~, num_targets] = size(b);
    out = utils.outInit();
    out.prim_hist = [];
    out.dual_hist = [];
    
    x = initial_x;
    sigma = opts.sigma;   % Quadratic penalty coefficient
    inverse_matrix = inv(eye(m) + sigma * A * A');
    z = zeros(n, num_targets);
    
    for outer_iter = 1:opts.maxit
        for inner_iter = 1:opts.maxit_inn
            % Update dual variable y
            y = inverse_matrix * (A * x - sigma * A * z - b);
            
            % Store previous z for convergence check
            previous_z = z;
            
            % Update primal variable z using the proximal operator
            z = updateZ(x / sigma - A' * y, mu);
            
            % Calculate the change in z for convergence
            z_change = norm(z - previous_z, 'fro');
            
            % Increment total inner iterations
            out.itr_inn = out.itr_inn + 1;
            
            % Check convergence of the inner loop
            if z_change < opts.thre_inn
                break;
            end
        end
        
        % Update primal variable x
        x = x - sigma * (A' * y + z);
        
        % Compute the primal objective function value
        primal_obj = utils.objFun(x, A, b, mu);
        
        % Compute the dual objective function value
        dual_obj = 0.5 * norm(y, 'fro')^2 - sum(sum(y .* b));
        
        % Print the difference between primal and dual objectives
        fprintf('primal_obj - dual_obj: %f\n', primal_obj - dual_obj);
        
        % Record history of objective values
        out.prim_hist = [out.prim_hist; primal_obj];
        out.dual_hist = [out.dual_hist; dual_obj];
        
        % Check convergence based on the norm of the residual
        if norm(A' * y + z, 'fro') < opts.thre
            break;
        end
    end
    
    % Store the number of iterations and final objective value
    out.itr = outer_iter;
    out.fval = out.prim_hist(end);
    fprintf('ALM_dual: itr: %d, fval: %f\n', out.itr, out.fval);
    fprintf('ALM_dual: itr_inn: %d\n', out.itr_inn);
    fprintf('ALM_dual: len(out.prim_hist): %d\n', length(out.prim_hist));
    
    % Prepare iteration history for output
    iters_range = 1:out.itr;
    out.iters = [iters_range', out.prim_hist];
    
    num_iters = out.itr;
end

function [z] = updateZ(reference, mu)
    % Update the variable Z using the proximal operator.
    %
    % Args:
    %   reference (matrix): Reference matrix for updating Z
    %   mu (double): Regularization parameter
    %
    % Returns:
    %   matrix: Updated Z matrix
    
    norm_reference = vecnorm(reference, 2, 2);  % Row-wise L2 norm
    norm_reference = reshape(norm_reference, [], 1);  % Ensure column vector
    norm_reference(norm_reference < mu) = mu;
    updated_z = reference .* (mu ./ norm_reference);
    z = updated_z;
end

