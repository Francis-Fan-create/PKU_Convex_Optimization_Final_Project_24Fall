function opts = ADMM_primal_optsInit(opts0)
    % Default parameters for ADMM primal method
    if nargin < 1 || isempty(opts0)
        opts0 = struct();
    end
    opts.sigma = get_field(opts0, 'sigma', 10);
    opts.maxit = get_field(opts0, 'maxit', 3000);
    opts.thre = get_field(opts0, 'thre', 1e-6);
end