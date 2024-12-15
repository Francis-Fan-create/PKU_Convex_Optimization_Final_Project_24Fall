function opts = ALM_dual_optsInit(opts0)
    % Default parameters for Augmented Lagrangian Method
    if nargin < 1 || isempty(opts0)
        opts0 = struct();
    end
    opts.sigma = get_field(opts0, 'sigma', 10);
    opts.maxit = get_field(opts0, 'maxit', 100);
    opts.maxit_inn = get_field(opts0, 'maxit_inn', 300);
    opts.thre = get_field(opts0, 'thre', 1e-6);
    opts.thre_inn = get_field(opts0, 'thre_inn', 1e-3);
end