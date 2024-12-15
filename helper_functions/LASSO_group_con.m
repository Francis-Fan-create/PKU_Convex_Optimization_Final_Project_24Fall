function [x, itr_inn, outResult] = LASSO_group_con(x0, A, b, mu0, opts)
    % Set default options if not provided
    opts.maxit = get_field(opts, 'maxit', 50);
    opts.etag = get_field(opts, 'etag', 0.1);
    opts.etaf = get_field(opts, 'etaf', 0.1);
    opts.gtol = get_field(opts, 'gtol', 1e-6);
    opts.gtol_init_ratio = get_field(opts, 'gtol_init_ratio', 1 / opts.gtol);
    opts.ftol = get_field(opts, 'ftol', 1e-9);
    opts.ftol_init_ratio = get_field(opts, 'ftol_init_ratio', 1e6);
    opts.factor = get_field(opts, 'factor', 0.1);
    opts.mu1 = get_field(opts, 'mu1', 10);
    opts.is_only_print_outer = get_field(opts, 'is_only_print_outer', false);
    opts.method = get_field(opts, 'method', []);
    % Compute eigenvalues
    eigs_values = eig(A' * A);
    eigs_values = real(eigs_values(isreal(eigs_values)));
    % Initialize parameters for inner loop
    optsInner = optsInnerInit(opts);
    optsInner.alpha0 = 1 / max(eigs_values);
    optsInner.mu0 = mu0;
    optsInner.ftol = opts.ftol * opts.ftol_init_ratio;
    optsInner.gtol = opts.gtol * opts.gtol_init_ratio;
    % Initialize result output
    outResult = outInit();
    x = x0;
    mu_t = opts.mu1;
    f = objFun(x, A, b, mu_t);
    solver = opts.method;
    for k = 1:opts.maxit
        % Tighten stopping criteria
        optsInner.gtol = max(optsInner.gtol * opts.etag, opts.gtol);
        optsInner.ftol = max(optsInner.ftol * opts.etaf, opts.ftol);
        % Start inner solver
        if ~isa(solver, 'function_handle')
            error('opts.method is not a valid function handle');
        end
        fp = f;
        [x, itr_inn, outInner] = solver(x, A, b, mu_t, optsInner);
        f = outInner.f_hist_inner(end);
        outResult.f_hist_inner = [outResult.f_hist_inner; outInner.f_hist_inner];
        outResult.f_hist_outer = [outResult.f_hist_outer; f];
        r = A * x - b;
        % Violation of the optimality condition
        nrmG = norm(x - prox(x - A' * r, mu0), 'fro');
        % Update mu_t if inner loop converged
        if outInner.flag
            mu_t = max(mu_t * opts.factor, mu0);
        end
        outResult.itr_inn = outResult.itr_inn + itr_inn;
        if mu_t == mu0 && (nrmG < opts.gtol || abs(f - fp) < opts.ftol)
            break;
        end
    end
    outResult.fval = f;  % Final objective function value
    outResult.iter = k;   % Outer loop k count
    % Determine whether to use only outer loop k info
    if opts.is_only_print_outer
        outResult.iters = [ (1:outResult.iter)', outResult.f_hist_outer ];
        itr_inn = outResult.iter;
    else
        outResult.iters = [ (1:outResult.itr_inn)', outResult.f_hist_inner ];
        itr_inn = outResult.itr_inn;
    end
end