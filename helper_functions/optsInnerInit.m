function optsInner = optsInnerInit(opts)
    % Initialize parameters for inner loop
    optsInner.mu0 = get_field(opts, 'mu0', 1e-2);
    optsInner.maxit_inn = get_field(opts, 'maxit_inn', 200);
    optsInner.ftol = get_field(opts, 'ftol', 1e-8);
    optsInner.gtol = get_field(opts, 'gtol', 1e-6);
    optsInner.alpha0 = get_field(opts, 'alpha0', 1);
    optsInner.gamma = get_field(opts, 'gamma', 0.9);
    optsInner.rhols = get_field(opts, 'rhols', 1e-6);
    optsInner.eta = get_field(opts, 'eta', 0.2);
    optsInner.Q = get_field(opts, 'Q', 1);
end