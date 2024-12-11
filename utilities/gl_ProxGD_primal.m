function [x, iterations, output] = gl_ProxGD_primal(initial_x, A, b, mu, opts)
    % Solve the group LASSO problem using the Proximal Gradient Method.
    if nargin < 5
        opts = struct();
    end
    opts = utils.ProxGD_primal_optsInit(opts);
    opts.method = @gl_ProxGD_primal_inner;
    [x, iterations, output] = utils.LASSO_group_con(initial_x, A, b, mu, opts);
end

function [x, num_iters, output] = gl_ProxGD_primal_inner(initial_x, A, b, mu, opts)
    % Inner function for the Fast Proximal Gradient Method to solve the group LASSO problem.
    if nargin < 5
        opts = struct();
    end
    opts = utils.optsInnerInit(opts);
    fprintf('--->optsInner:<---\n'); disp(opts);
    
    target_mu = opts.mu0;  % Target minimum mu0
    step_size = opts.alpha0;  % Initial step size
    
    % Initial computation
    x = initial_x;
    y = x;
    previous_x = initial_x;
    gradient = A' * (A * y - b);
    temp = 0.5 * norm(A * y - b, 'fro')^2;
    current_val = temp + mu * sum(vecnorm(x, 2, 2));
    f_val = temp + target_mu * sum(vecnorm(x, 2, 2));
    gradient_norm = norm(x - utils.prox(x - gradient, mu), 'fro');
    
    output = utils.outInit();
    best_f_val = 1e7;
    
    for iteration = 0:opts.maxit_inn-1
        previous_y = y;
        previous_gradient = gradient;
        previous_x = x;
        
        output.g_hist = [output.g_hist; gradient_norm];
        output.f_hist_inner = [output.f_hist_inner; f_val];
        best_f_val = min(best_f_val, f_val);
        
        output.f_hist_best = [output.f_hist_best; best_f_val];
        
        fprintf('  inner iter %d: fval: %f, f_best: %f\n', iteration, f_val, best_f_val);
        fprintf('\tabs(fval - f_best) = %f\n', abs(f_val - best_f_val));
        fprintf('\topts.ftol = %f\n', opts.ftol);
        if iteration > 0
            fprintf('\tout.f_hist_inner(%d) - out.f_hist_inner(%d) = %f\n', ...
                iteration+1, iteration, output.f_hist_inner(end) - output.f_hist_inner(end-1));
        end
        fprintf('\topts.gtol = %f\n', opts.gtol);
        fprintf('\tout.g_hist(%d) = %f\n', iteration+1, output.g_hist(end));
        
        if iteration > 2 && ...
           abs(output.f_hist_inner(end) - output.f_hist_inner(end-1)) < opts.ftol && ...
           output.g_hist(end) < opts.gtol
            output.flag = true;
            break;
        end
        
        % Line search
        for nls_attempt = 1:10
            x_candidate = utils.prox(previous_x - step_size * gradient, step_size * mu);
            temp_candidate = 0.5 * norm(A * x_candidate - b, 'fro')^2;
            f_candidate = temp_candidate + mu * sum(vecnorm(x_candidate, 2, 2));
            
            if f_candidate <= current_val - 0.5 * step_size * opts.rhols * gradient_norm^2
                x = x_candidate;
                break;
            end
            step_size = step_size * opts.eta;
        end
        
        theta = (iteration - 1) / (iteration + 2);  % iteration starts at 0
        y = x + theta * (x - previous_x);
        residual = A * y - b;
        gradient = A' * residual;
        f_val = temp_candidate + target_mu * sum(vecnorm(x, 2, 2));
        
        % Barzilai-Borwein (BB) step size update
        step_size = utils.BBupdate(y, previous_y, gradient, previous_gradient, iteration, step_size);
        
        gradient_norm = norm(x - y, 'fro') / step_size;
        previous_Q = opts.Q;
        opts.Q = opts.gamma * previous_Q + 1;
        current_val = (opts.gamma * previous_Q * current_val + f_candidate) / opts.Q;
    end
    
    output.itr = iteration + 1;
    output.flag = true;
end