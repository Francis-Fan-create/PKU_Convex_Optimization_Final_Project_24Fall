function [x, iterations, output] = gl_SGD_primal(initial_x, A, b, mu, opts)
    % Subgradient method for the primal problem.
    % - Minimize (1/2)*||Ax - b||_2^2 + mu*||x||_{1,2}
    % - A is m x n, b is m x l, x is n x l
    if nargin < 5
        opts = struct();
    end
    opts = utils.SGD_primal_optsInit(opts);
    opts.method = @gl_SGD_primal_inner;
    [x, iterations, output] = utils.LASSO_group_con(initial_x, A, b, mu, opts);
end

function [x, num_iters, output] = gl_SGD_primal_inner(initial_x, A, b, mu, opts)
    % Inner function for the Subgradient Method to solve the group LASSO problem.
    if nargin < 5
        opts = struct();
    end
    opts = utils.optsInnerInit(opts);
    fprintf('--->optsInner:<---\n'); disp(opts);
    
    target_mu = opts.mu0;  % Target minimum mu0
    step_size = opts.alpha0;  % Initial step size
    
    % Initial computation
    current_x = initial_x;
    residual = A * current_x - b;
    gradient = A' * residual;
    norm_x = vecnorm(current_x, 2, 2);
    norm_x = reshape(norm_x, [], 1);
    subgradient = current_x ./ ((norm_x <= 1e-6) + norm_x);
    subgradient = subgradient * mu + gradient;
    gradient_norm = norm(gradient, 'fro');
    temp_value = 0.5 * norm(residual, 'fro')^2;
    current_val = temp_value + mu * sum(vecnorm(current_x, 2, 2));
    function_val = temp_value + target_mu * sum(vecnorm(current_x, 2, 2));
    fprintf('inner function value: %f\n', function_val);
    
    output = utils.outInit();
    
    for iteration = 0:opts.maxit_inn-1
        previous_gradient = gradient;
        previous_x = current_x;
        
        output.g_hist = [output.g_hist; gradient_norm];
        output.f_hist_inner = [output.f_hist_inner; function_val];
        
        % Check for convergence
        if iteration > 2 && abs(output.f_hist_inner(end) - output.f_hist_inner(end-1)) < opts.ftol
            output.flag = true;
            break;
        end
        
        % Line search
        for nls_attempt = 1:10
            candidate_x = utils.prox(previous_x - step_size * subgradient, step_size * mu);
            candidate_residual = 0.5 * norm(A * candidate_x - b, 'fro')^2;
            candidate_f = candidate_residual + mu * sum(vecnorm(candidate_x, 2, 2));
            
            if candidate_f <= current_val - 0.5 * step_size * opts.rhols * gradient_norm^2
                current_x = candidate_x;
                break;
            end
            step_size = step_size * opts.eta;
        end
        
        % Update gradients after x update
        gradient = A' * (A * current_x - b);
        norm_x = vecnorm(current_x, 2, 2);
        norm_x = reshape(norm_x, [], 1);
        subgradient = current_x ./ ((norm_x <= 1e-6) + norm_x);
        subgradient = subgradient * mu + gradient;
        
        % Update function value
        function_val = candidate_residual + target_mu * sum(vecnorm(current_x, 2, 2));
        
        % Update gradient norm
        gradient_norm = norm(gradient, 'fro');
        
        previous_Q = opts.Q;
        opts.Q = opts.gamma * previous_Q + 1;
        current_val = (opts.gamma * previous_Q * current_val + candidate_f) / opts.Q;
        
        % Barzilai-Borwein (BB) step size update
        step_size = utils.BBupdate(current_x, previous_x, gradient, previous_gradient, iteration, step_size);
    end
    
    output.itr = iteration + 1;
end