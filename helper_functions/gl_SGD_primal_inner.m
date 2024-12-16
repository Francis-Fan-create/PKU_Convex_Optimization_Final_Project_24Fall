function [x, iter, output] = gl_SGD_primal_inner(x0, A, b, mu, opts)
    % Inner function for the sub_g Method to solve the group LASSO problem.
    opts = optsInnerInit(opts);
    
    mu0 = opts.mu0;  % Target minimum mu0
    alpha = opts.alpha0;  % Initial step size
    
    % Initial computation
    x = x0;
    r = A * x - b;
    g = A' * r;
    norm_x = vecnorm(x, 2, 2);
    sub_g = x ./ ((norm_x <= 1e-6) + norm_x);
    sub_g = sub_g * mu + g;
    nrmG = norm(g, 'fro');
    tmp = 0.5 * norm(r, 'fro').^2;
    Cval = tmp + mu .* sum(vecnorm(x, 2, 2));
    f = tmp + mu0 .* sum(vecnorm(x, 2, 2));
    
    output = outInit();
    
    for k = 1:opts.maxit_inn
        gp = g;
        xp = x;
        
        output.g_hist = [output.g_hist; nrmG];
        output.f_hist_inner = [output.f_hist_inner; f];
        
        % Check for convergence
        if k > 3 && abs(output.f_hist_inner(k) - output.f_hist_inner(k-1)) < opts.ftol
            output.flag = true;
            break;
        end
        
        % Line search
        for nls = 1:10
            x = xp - alpha .* sub_g;
            tmp = 0.5 .* norm(A * x - b, 'fro').^2;
            tmpf = tmp + mu .* sum(vecnorm(x, 2, 2));
            
            if tmpf <= Cval - 0.5 .* alpha .* opts.rhols * nrmG.^2
                break;
            end
            alpha = alpha .* opts.eta;
        end
        
        % Update gs after x update
        g = A' * (A * x - b);
        norm_x = vecnorm(x, 2, 2);
        sub_g = x ./ ((norm_x <= 1e-6) + norm_x);
        sub_g = sub_g .* mu + g;
        
        % Update function value
        f = tmp + mu0 .* sum(vecnorm(x, 2, 2));
        
        % Update g norm
        nrmG = norm(g, 'fro');
        
        Qp = opts.Q;
        opts.Q = opts.gamma .* Qp + 1;
        Cval = (opts.gamma .* Qp .* Cval + tmpf) ./ opts.Q;
        
        % Barzilai-Borwein (BB) step size update
        alpha = BBupdate(x, xp, g, gp, k-1, alpha);
    end
    
    output.iter = k ;
    iter = k ;
end
    
