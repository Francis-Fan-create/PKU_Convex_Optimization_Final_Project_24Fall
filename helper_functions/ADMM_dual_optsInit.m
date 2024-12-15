function opts = ADMM_dual_optsInit(opts0)
    % Default parameters for ADMM dual method
    if nargin < 1 || isempty(opts0)
        opts0 = struct();
    end
    opts.sigma = get_field(opts0, 'sigma', 10);
    opts.maxit = get_field(opts0, 'maxit', 1000);
    opts.thre = get_field(opts0, 'thre', 1e-6);
end