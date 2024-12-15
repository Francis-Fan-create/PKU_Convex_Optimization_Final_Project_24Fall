function [x, iter, out] = gl_FProxGD_primal(x0, A, b, mu, opts)
    % Fast Proximal g Method for Group LASSO Problem
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
    %     iter (int): Number of ks.
    %     out (struct): Output information.

    if nargin < 5 || isempty(opts)
        opts = struct();
    end
    opts = FProxGD_primal_optsInit(opts);
    opts.method = @gl_FProxGD_primal_inner;
    [x, iter, out] = LASSO_group_con(x0, A, b, mu, opts);
end

function [x, iter, out] = gl_FProxGD_primal_inner(x0, A, b, mu, opts)
    % Fast Proximal g Method Inner Loop for Group LASSO Problem
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
    %     iter (int): Number of ks.
    %     out (struct): Output information.

    % Initialize inner loop options
    opts = optsInnerInit(opts);

    mu0 = opts.mu0;     % Target minimum mu0
    alpha = opts.alpha0;  % Initial step size

    % Initialize variables
    x = x0;
    y = x;
    xp = x0;
    g = A' * (A * y - b);
    tmp = 0.5 * norm(A * y - b, 'fro')^2;
    Cval = tmp + mu * sum(vecnorm(x, 2, 2));
    f = tmp + mu0 * sum(vecnorm(x, 2, 2));
    nrmG = norm(x - prox(x - g, mu), 'fro');

    % Initialize output structure
    out = outInit();
    best_f = Inf;

    for k = 1:opts.maxit_inn
        yp = y;
        gp = g;
        xp = x;

        out.g_hist = [out.g_hist; nrmG];
        out.f_hist_inner = [out.f_hist_inner; f];
        best_f = min(best_f, f);
        out.f_hist_best = [out.f_hist_best; best_f];

        if k > 3 && abs(out.f_hist_inner(end) - out.f_hist_inner(end - 1)) < opts.ftol && out.g_hist(end) < opts.gtol
            out.flag = true;
            break;
        end

        for nls = 1:10
            x = prox(y - alpha * g, alpha * mu);
            tmp = 0.5 * norm(A * x - b, 'fro')^2;
            tmpf = tmp + mu * sum(vecnorm(x, 2, 2));

            if tmpf <= Cval - 0.5 * alpha * opts.rhols * nrmG^2
                break;
            end
            alpha = alpha * opts.eta;
        end

        theta = (k - 2) / (k + 1);  % iter starts from 1
        y = x + theta * (x - xp);
        r = A * y - b;
        g = A' * r;
        f = tmp + mu0 * sum(vecnorm(x, 2, 2));

        % Barzilai-Borwein (BB) step size update
        alpha = BBupdate(y, yp, g, gp, k, alpha);

        nrmG = norm(x - y, 'fro') / alpha;
        Qp = opts.Q;
        opts.Q = opts.gamma * Qp + 1;
        Cval = (opts.gamma * Qp * Cval + tmpf) / opts.Q;
    end

    out.iter = k;
    iter = k;
    out.flag = true;
end

