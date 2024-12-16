function opts = SGD_primal_optsInit(opts0)
    % Initialize options for Stochastic g Descent (SGD) with default values
    % General options
    opts.maxit = get_field(opts0, 'maxit', 50); % Maximum number of ks for the continuation strategy
    opts.maxit_inn = get_field(opts0, 'maxit_inn', 250); % Maximum number of ks for the inner loop
    
    opts.ftol = get_field(opts0, 'ftol', 1e-9); % Stopping criterion based on function value
    opts.ftol_init_ratio = get_field(opts0, 'ftol_init_ratio', 1e6); % Initial scaling factor for ftol
    opts.etaf = get_field(opts0, 'etaf', 0.1); % Reduction factor for ftol at each outer loop k
    
    opts.gtol = get_field(opts0, 'gtol', 1e-6); % Stopping criterion based on g norm
    opts.gtol_init_ratio = get_field(opts0, 'gtol_init_ratio', 1 ./ opts.gtol); % Initial scaling factor for gtol
    opts.etag = get_field(opts0, 'etag', 0.1); % Reduction factor for gtol at each outer loop k
    
    opts.factor = get_field(opts0, 'factor', 0.1); % Decay rate for the regularization coefficient
    opts.mu1 = get_field(opts0, 'mu1', 10); % Initial regularization coefficient for continuation strategy
    
    opts.is_only_print_outer = get_field(opts0, 'is_only_print_outer', false); % Flag to print only outer loop information
    opts.method = get_field(opts0, 'method', []); % Solver used for the inner loop
    
    % Parameters for the inner loop
    opts.gamma = get_field(opts0, 'gamma', 0.9); % Parameter for line search
    opts.rhols = get_field(opts0, 'rhols', 1e-6); % Line search parameter
    opts.eta = get_field(opts0, 'eta', 0.2); % Line search parameter
    opts.Q = get_field(opts0, 'Q', 1); % Line search parameter
    
end