% utils.m
classdef utils
    properties(Constant)
        solversCollection = {...
            'gl_cvx_mosek', ...
            'gl_cvx_gurobi', ...
            'gl_mosek', ...
            'gl_gurobi', ...
            'gl_SGD_primal', ...
            'gl_ProxGD_primal', ...
            'gl_FProxGD_primal', ...
            'gl_ALM_dual', ...
            'gl_ADMM_dual', ...
            'gl_ADMM_primal' ...
        };
    end

    methods(Static)
        function [logger, loggerName, cvxLogsName] = loggerInit(name)
            % Initialize logging for test files
            %
            % Args:
            %   name: Name of the test file
            %
            % Returns:
            %   logger: Logger object (empty in MATLAB implementation) 
            %   loggerName: Full path to the log file
            %   cvxLogsName: Full path to CVX solver log file
            
            currentWorkDir = fileparts(mfilename('fullpath'));
            logDir = fullfile(currentWorkDir, 'logs');
            
            % Create logs directory if it doesn't exist
            if ~exist(logDir, 'dir')
                mkdir(logDir);
            end
            
            % Generate timestamp for log file
            timestamp = datestr(now, 'mmdd-HHMMSS');
            
            % Create log file paths
            loggerName = fullfile(logDir, sprintf('%s-%s.log', name, timestamp));
            cvxLogsName = fullfile(logDir, 'gl_cvx.log');
            
            % Start logging to file
            diary(loggerName);
            
            % Return empty logger since MATLAB uses diary
            logger = [];
        end
        
        function results = parse_iters(log_str, solver)
            % Using regexp instead of re.compile
            re_iterc_default = '^\s*(?<iterc>\d{1,3})\:?\s+(?<objv>[0-9\.eE\+\-]+)';
            reg_solver = containers.Map();
            reg_solver('GUROBI') = re_iterc_default;
            reg_solver('MOSEK') = '^\s*([\s\S]{26})\:(\s+)(?<iterc>\d{1,2})\s([\s\S]{38})(\s+)(?<objv>[\-\+0-9\.eE]+)';
            reg_solver('MOSEK_OLD') = '^\s*(?<iterc>\d{1,3})\:?(\s+(?:[0-9\.eE\+\-]+)){4}\s+(?<objv>[0-9\.eE\+\-]+)';
            reg_solver('CVXOPT') = re_iterc_default;
            
            if isKey(reg_solver, solver)
                pattern = reg_solver(solver);
            else
                pattern = re_iterc_default;
            end
            
            matches = regexp(log_str, pattern, 'names');
            results = cell(length(matches), 2);
            for i = 1:length(matches)
                results{i,1} = str2double(matches(i).iterc);
                results{i,2} = str2double(matches(i).objv);
            end
        end
        
        function cleanUpLog()
            fid = fopen(cvxLogsName, 'w');
            fclose(fid);
        end
        
        function s = sparsity(x)
            s = sum(abs(x) > 1e-5, 'all') / numel(x);
        end
        
        function err = errX(x, x0)
            err = norm(x - x0, 'fro') / (1 + norm(x0, 'fro'));
        end
        
        function obj = objFun(x, A, b, mu)
            obj = 0.5 * norm(A * x - b, 'fro')^2 + mu * sum(vecnorm(x, 2, 2));
        end
        
        function err = errObj(obj, obj0)
            err = abs(obj - obj0);
        end
        
        function prox_x = prox(x, mu)
            norm_x = vecnorm(x, 2, 2);
            mask = norm_x > mu;
            prox_x = x - mu * x ./ (norm_x + 1e-10);
            prox_x = prox_x .* mask;
        end
        
        function alpha = BBupdate(x, x_prev, grad, grad_prev, k, alpha)
            dx = x - x_prev;
            dg = grad - grad_prev;
            dxg = abs(sum(dx .* dg, 'all'));
            if dxg > 1e-12
                if mod(k, 2) == 1
                    alpha = sum(dx .* dx, 'all') / dxg;
                else
                    alpha = dxg / sum(dg .* dg, 'all');
                end
            end
            alpha = max(min(alpha, 1e12), 1e-12);
        end
        
        function opts = testDataParams(opts0)
            if isempty(opts0)
                opts0 = struct();
            end
            opts = struct();
            opts.seed = getfield_default(opts0, 'seed', 97108120);
            opts.mu = getfield_default(opts0, 'mu', 1e-2);
            opts.n = getfield_default(opts0, 'n', 512);
            opts.m = getfield_default(opts0, 'm', 256);
            opts.l = getfield_default(opts0, 'l', 2);
            opts.r = getfield_default(opts0, 'r', 1e-1);
        end
        
        function [x0, A, b, mu, u, f_u] = testData(opts)
            % Generate test data for group LASSO problem
            %
            % Args:
            %   opts: struct with fields:
            %     seed: random seed
            %     mu: regularization parameter
            %     n: number of features
            %     m: number of samples
            %     l: number of tasks
            %     r: sparsity ratio
            %
            % Returns:
            %   x0: initial guess
            %   A: constraint matrix
            %   b: observation vector
            %   mu: regularization parameter
            %   u: exact solution
            %   f_u: objective value at exact solution
            
            if isempty(opts)
                opts = utils.testDataParams(struct());
            end
            
            % Set random seed
            rng(opts.seed);
            
            % Generate random data
            A = randn(opts.m, opts.n);
            k = round(opts.n * opts.r);
            indices = randperm(opts.n, k);
            u = zeros(opts.n, opts.l);
            u(indices, :) = randn(k, opts.l);
            b = A * u;
            x0 = randn(opts.n, opts.l);
            mu = opts.mu;
            f_u = utils.objFun(u, A, b, mu);
        end
        
        function [x, num_iters, out] = testSolver(x0, A, b, mu, opts)
            if ~isfield(opts, 'solver_name')
                error('The opts struct must contain the field "solver_name"');
            end
            solver_name = opts.solver_name;
            
            try
                solver = str2func(solver_name);
            catch
                warning('Solver %s does not exist, skipping this solver.', solver_name);
                x = []; num_iters = []; out = [];
                return;
            end
            
            fprintf('\n--->Current Test Solver: %s<---\n', solver_name);
            solver_opts = getfield_default(opts, solver_name(4:end), struct());
            
            tic;
            [x, num_iters, out] = solver(x0, A, b, mu, solver_opts);
            time_cpu = toc;
            
            utils.cleanUpLog();
            out.time_cpu = time_cpu;
            sparsity_x = utils.sparsity(x);
            out.sparsity_x = sparsity_x;
            fprintf('%s takes %.5fs, with %d iterations\n', solver_name(4:end), time_cpu, num_iters);
        end

                function opts = SGD_primal_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.maxit = getfield_default(opts0, 'maxit', 50);  % Maximum iterations for the continuation strategy
            opts.maxit_inn = getfield_default(opts0, 'maxit_inn', 250);  % Maximum iterations for the inner loop
            
            opts.ftol = getfield_default(opts0, 'ftol', 1e-9);  % Stopping criterion for function value
            opts.ftol_init_ratio = getfield_default(opts0, 'ftol_init_ratio', 1e6);  % Initial ratio for ftol
            opts.etaf = getfield_default(opts0, 'etaf', 0.1);  % Reduction factor for ftol in each outer iteration
            
            opts.gtol = getfield_default(opts0, 'gtol', 1e-6);  % Stopping criterion for gradient norm
            opts.gtol_init_ratio = getfield_default(opts0, 'gtol_init_ratio', 1/opts.gtol);  % Initial ratio for gtol
            opts.etag = getfield_default(opts0, 'etag', 0.1);  % Reduction factor for gtol in each outer iteration
            
            opts.factor = getfield_default(opts0, 'factor', 0.1);  % Decay rate for regularization parameter
            opts.mu1 = getfield_default(opts0, 'mu1', 10);  % Initial regularization parameter for continuation strategy
            
            opts.is_only_print_outer = getfield_default(opts0, 'is_only_print_outer', false);  % Whether to print only outer loop info
            opts.method = getfield_default(opts0, 'method', []);  % Solver used in inner loop
            
            % Parameters for inner loop
            opts.gamma = getfield_default(opts0, 'gamma', 0.9);
            opts.rhols = getfield_default(opts0, 'rhols', 1e-6);  % Line search parameter
            opts.eta = getfield_default(opts0, 'eta', 0.2);  % Line search parameter
            opts.Q = getfield_default(opts0, 'Q', 1);  % Line search parameter
        end
        
        function opts = ProxGD_primal_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.maxit = getfield_default(opts0, 'maxit', 50);
            opts.maxit_inn = getfield_default(opts0, 'maxit_inn', 250);
            
            opts.ftol = getfield_default(opts0, 'ftol', 1e-9);
            opts.ftol_init_ratio = getfield_default(opts0, 'ftol_init_ratio', 1e6);
            opts.etaf = getfield_default(opts0, 'etaf', 0.1);
            
            opts.gtol = getfield_default(opts0, 'gtol', 1e-6);
            opts.gtol_init_ratio = getfield_default(opts0, 'gtol_init_ratio', 1/opts.gtol);
            opts.etag = getfield_default(opts0, 'etag', 0.1);
            
            opts.factor = getfield_default(opts0, 'factor', 0.1);
            opts.mu1 = getfield_default(opts0, 'mu1', 10);
            
            opts.is_only_print_outer = getfield_default(opts0, 'is_only_print_outer', false);
            opts.method = getfield_default(opts0, 'method', []);
            
            opts.gamma = getfield_default(opts0, 'gamma', 0.85);
            opts.rhols = getfield_default(opts0, 'rhols', 1e-6);
            opts.eta = getfield_default(opts0, 'eta', 0.2);
            opts.Q = getfield_default(opts0, 'Q', 1);
        end
        
        function opts = FProxGD_primal_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.maxit = getfield_default(opts0, 'maxit', 50);
            opts.maxit_inn = getfield_default(opts0, 'maxit_inn', 250);
            
            opts.ftol = getfield_default(opts0, 'ftol', 1e-9);
            opts.ftol_init_ratio = getfield_default(opts0, 'ftol_init_ratio', 1e6);
            opts.etaf = getfield_default(opts0, 'etaf', 0.1);
            
            opts.gtol = getfield_default(opts0, 'gtol', 1e-6);
            opts.gtol_init_ratio = getfield_default(opts0, 'gtol_init_ratio', 1/opts.gtol);
            opts.etag = getfield_default(opts0, 'etag', 0.1);
            
            opts.factor = getfield_default(opts0, 'factor', 0.1);
            opts.mu1 = getfield_default(opts0, 'mu1', 10);
            
            opts.is_only_print_outer = getfield_default(opts0, 'is_only_print_outer', false);
            opts.method = getfield_default(opts0, 'method', []);
            
            opts.gamma = getfield_default(opts0, 'gamma', 0.85);
            opts.rhols = getfield_default(opts0, 'rhols', 1e-6);
            opts.eta = getfield_default(opts0, 'eta', 0.2);
            opts.Q = getfield_default(opts0, 'Q', 1);
        end
        
        function opts = ALM_dual_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.sigma = getfield_default(opts0, 'sigma', 10);
            opts.maxit = getfield_default(opts0, 'maxit', 100);
            opts.maxit_inn = getfield_default(opts0, 'maxit_inn', 300);
            opts.thre = getfield_default(opts0, 'thre', 1e-6);
            opts.thre_inn = getfield_default(opts0, 'thre_inn', 1e-3);
        end
        
        function opts = ADMM_dual_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.sigma = getfield_default(opts0, 'sigma', 10);
            opts.maxit = getfield_default(opts0, 'maxit', 1000);
            opts.thre = getfield_default(opts0, 'thre', 1e-6);
        end
        
        function opts = ADMM_primal_optsInit(opts0)
            if nargin < 1
                opts0 = struct();
            end
            opts = struct();
            opts.sigma = getfield_default(opts0, 'sigma', 10);
            opts.maxit = getfield_default(opts0, 'maxit', 3000);
            opts.thre = getfield_default(opts0, 'thre', 1e-6);
        end
        
        function optsInner = optsInnerInit(opts)
            if nargin < 1
                opts = struct();
            end
            optsInner = struct();
            optsInner.mu0 = getfield_default(opts, 'mu0', 1e-2);  % Target minimum mu0
            optsInner.maxit_inn = getfield_default(opts, 'maxit_inn', 200);
            optsInner.ftol = getfield_default(opts, 'ftol', 1e-8);
            optsInner.gtol = getfield_default(opts, 'gtol', 1e-6);
            optsInner.alpha0 = getfield_default(opts, 'alpha0', 1);
            optsInner.gamma = getfield_default(opts, 'gamma', 0.9);
            optsInner.rhols = getfield_default(opts, 'rhols', 1e-6);
            optsInner.eta = getfield_default(opts, 'eta', 0.2);
            optsInner.Q = getfield_default(opts, 'Q', 1);
        end
        
        function printAllDefaultOpts()
            fprintf('testData: %s\n', utils.testDataParams(struct()));
            fprintf('gl_SGD_primal: %s\n', utils.SGD_primal_optsInit(struct()));
            fprintf('gl_ProxGD_primal: %s\n', utils.ProxGD_primal_optsInit(struct()));
            fprintf('gl_FProxGD_primal: %s\n', utils.FProxGD_primal_optsInit(struct()));
            fprintf('gl_ALM_dual: %s\n', utils.ALM_dual_optsInit(struct()));
            fprintf('gl_ADMM_dual: %s\n', utils.ADMM_dual_optsInit(struct()));
            fprintf('gl_ADMM_primal: %s\n', utils.ADMM_primal_optsInit(struct()));
        end
        
        function out = outInit()
            out = struct();
            out.f_hist_outer = [];  % Objective function values in each outer iteration
            out.f_hist_inner = [];  % Objective function values in each inner iteration
            out.f_hist_best = [];   % Best objective function values in each inner iteration
            out.g_hist = [];        % History of gradient norms
            out.itr = 0;            % Outer loop iteration count
            out.itr_inn = 0;        % Total inner loop iteration count
            out.iters = [];         % Iterator for function values
            out.fval = 0;           % Final objective function value
            out.flag = false;       % Convergence flag
        end

        function [x, num_iters, outResult] = LASSO_group_con(x0, A, b, mu0, opts)
            if nargin < 5
                opts = struct();
            end
            
            % Get eigenvalues
            eigs_vals = eig(A' * A);
            eigs_vals = real(eigs_vals(imag(eigs_vals) == 0));
            
            % Initialize parameters for inner loop
            optsInner = utils.optsInnerInit(opts);
            optsInner.alpha0 = 1 / max(eigs_vals);
            optsInner.mu0 = mu0;
            optsInner.ftol = opts.ftol * opts.ftol_init_ratio;
            optsInner.gtol = opts.gtol * opts.gtol_init_ratio;
            
            % Initialize result output
            outResult = utils.outInit();
            
            x = x0;
            mu_t = opts.mu1;
            f = utils.objFun(x, A, b, mu_t);
            solver = opts.method;
            
            fprintf('solver: %s\n', func2str(solver));
            fprintf('optsOuter:\n'); disp(opts);
            fprintf('optsInner:\n'); disp(optsInner);
            
            for k = 1:opts.maxit
                fprintf('--->iter %d : current mu_t: %f<---\n', k, mu_t);
                fprintf('current fval: %f\n', f);
                fprintf('current alpha0: %f\n', optsInner.alpha0);
                
                % Tighten stopping criteria
                optsInner.gtol = max(optsInner.gtol * opts.etag, opts.gtol);
                optsInner.ftol = max(optsInner.ftol * opts.etaf, opts.ftol);
                fprintf('optsInner.ftol: %f\n', optsInner.ftol);
                fprintf('optsInner.gtol: %f\n', optsInner.gtol);
                
                % Start inner solver
                if ~isa(solver, 'function_handle')
                    error('opts.method is not a function handle');
                end
                fp = f;
                [x, itr_inn, outInner] = solver(x, A, b, mu_t, optsInner);
                f = outInner.f_hist_inner(end);
                outResult.f_hist_inner = [outResult.f_hist_inner; outInner.f_hist_inner];
                outResult.f_hist_outer = [outResult.f_hist_outer; f];
                
                r = A * x - b;
                % Since L1-norm is non-differentiable, nrmG represents the violation 
                % of the optimality condition of the LASSO problem
                nrmG = norm(x - utils.prox(x - A' * r, mu0), 'fro');
                fprintf('current nrmG: %f\n', nrmG);
                fprintf('current abs(f-fp): %f\n', abs(f - fp));
                fprintf('current itr_inn: %d\n', itr_inn);
                fprintf('is_inner_converged: %d\n', outInner.flag);
                
                % By default, mu_t is reduced in each outer iteration
                % If inner loop converges within specified iterations, don't reduce mu_t
                if outInner.flag
                    mu_t = max(mu_t * opts.factor, mu0);
                end
                
                outResult.itr_inn = outResult.itr_inn + itr_inn;
                
                if mu_t == mu0 && (nrmG < opts.gtol || abs(f - fp) < opts.ftol)
                    fprintf('--->fval has converged to %f\n', f);
                    fprintf('--->nrmG has converged to %f\n', nrmG);
                    fprintf('--->abs(f - fp) has converged to %f\n', abs(f - fp));
                    break;
                end
            end
            
            outResult.fval = f;
            outResult.itr = k;
            fprintf('len(outResult.f_hist_inner): %d\n', length(outResult.f_hist_inner));
            fprintf('outResult.itr_inn: %d\n', outResult.itr_inn);
            fprintf('--->end of LASSO_group_con<---\n');
            
            % Determine whether to use only outer loop iteration info
            if opts.is_only_print_outer
                iters_range = 1:outResult.itr;
                outResult.iters = [iters_range', outResult.f_hist_outer];
                num_iters = outResult.itr;
            else
                iters_range = 1:outResult.itr_inn;
                outResult.iters = [iters_range', outResult.f_hist_inner];
                num_iters = outResult.itr_inn;
            end  
        end  
    end
end

% Helper function for struct field access with default value
function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

