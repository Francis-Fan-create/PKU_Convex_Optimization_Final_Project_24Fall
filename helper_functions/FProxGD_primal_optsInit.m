function opts = FProxGD_primal_optsInit(opts0)
    % Default parameters for Fast Proximal g Descent method
    if nargin < 1 || isempty(opts0)
        opts0 = struct();
    end
    opts.maxit = get_field(opts0, 'maxit', 50);
    opts.maxit_inn = get_field(opts0, 'maxit_inn', 250);
    opts.ftol = get_field(opts0, 'ftol', 1e-9);
    opts.ftol_init_ratio = get_field(opts0, 'ftol_init_ratio', 1e6);
    opts.etaf = get_field(opts0, 'etaf', 0.1);
    opts.gtol = get_field(opts0, 'gtol', 1e-6);
    opts.gtol_init_ratio = get_field(opts0, 'gtol_init_ratio', 1 / opts.gtol);
    opts.etag = get_field(opts0, 'etag', 0.1);
    opts.factor = get_field(opts0, 'factor', 0.1);
    opts.mu1 = get_field(opts0, 'mu1', 10);
    opts.is_only_print_outer = get_field(opts0, 'is_only_print_outer', false);
    opts.method = get_field(opts0, 'method', []);
    opts.gamma = get_field(opts0, 'gamma', 0.85);
    opts.rhols = get_field(opts0, 'rhols', 1e-6);
    opts.eta = get_field(opts0, 'eta', 0.2);
    opts.Q = get_field(opts0, 'Q', 1);
end