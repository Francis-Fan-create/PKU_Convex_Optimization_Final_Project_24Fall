function Test_group_lasso(varargin)
    % Add utilities folder to path
    currentDir = fileparts(mfilename('fullpath'));
    utilitiesDir = fullfile(currentDir, 'utilities');
    if ~isfolder(utilitiesDir)
        error('Utilities folder not found. Expected at: %s', utilitiesDir);
    end
    addpath(utilitiesDir);
    
    % Check and initialize CVX
    if ~exist('cvx_begin', 'file')
        cvxDir = fullfile(currentDir, 'cvx'); % Adjust path as needed
        if ~isfolder(cvxDir)
            error(['CVX not found. Please install CVX from http://cvxr.com/cvx/ \n' ...
                   'and place it in: %s'], cvxDir);
        end
        run(fullfile(cvxDir, 'cvx_setup.m'));
    end
    
    % Parse input parameters
    p = inputParser;
    addParameter(p, 'solvers', {'gl_cvx_gurobi', 'gl_cvx_mosek'}, @iscell);
    addParameter(p, 'seed', 97006855, @isnumeric);
    addParameter(p, 'plot', false, @islogical);
    addParameter(p, 'log', 'INFO', @ischar);
    addParameter(p, 'opts', struct(), @isstruct);
    addParameter(p, 'compare', false, @islogical);
    addParameter(p, 'printDefaultOpts', false, @islogical);
    parse(p, varargin{:});
    args = p.Results;

    % Initialize logger
    [~, loggerName, cvxLogsName] = utils.loggerInit('test_group_lasso');

    % Handle printDefaultOpts
    if args.printDefaultOpts
        utils.printAllDefaultOpts();
        return;
    end

    % Process solvers parameter
    if length(args.solvers) == 1 && strcmpi(args.solvers{1}, 'all')
        args.solvers = utils.solversCollection;
    end

    % Print input parameters
    fprintf('Test solvers: %s\n', strjoin(args.solvers, ', '));
    fprintf('opts: %s\n', struct2str(args.opts));
    fprintf('random seed: %d\n', args.seed);
    fprintf('Is plotting?: %d\n', args.plot);
    fprintf('Is comparing?: %d\n', args.compare);
    fprintf('log level: %s\n', args.log);

    % Initialize test data
    data_options = struct();
    if isfield(args.opts, 'testData')
        data_options = args.opts.testData;
    end
    % Get default parameters and merge with user options
    data_options = utils.testDataParams(data_options);
    data_options.seed = args.seed;  % Override seed with command line argument
    
    % Now call testData with properly initialized options
    [x0, A, b, mu, u, f_u] = utils.testData(data_options);
    sparsity_u = utils.sparsity(u);
    
    fprintf('Objective function value of exact solution f_u: %f\n', f_u);
    fprintf('Sparsity of exact solution sparsity_u: %f\n', sparsity_u);
    fprintf('\n#######==Start all solvers TEST==#######\n');

    % Initialize results table
    table_data = {};

    % Redirect output (Note: MATLAB doesn't have direct equivalent, using diary)
    diary(cvxLogsName);

    % Compare solvers if requested
    if args.compare
        fprintf('\n--->Compare Solver: cvx_mosek and cvx_gurobi<---');
        [x_mosek, iters_N_mosek, out_mosek] = utils.testSolver(x0, A, b, mu, struct('solver_name', 'gl_cvx_mosek'));
        [x_gurobi, iters_N_gurobi, out_gurobi] = utils.testSolver(x0, A, b, mu, struct('solver_name', 'gl_cvx_gurobi'));
        
        table_data{end+1} = {'cvx_mosek', out_mosek.fval, utils.errObj(out_mosek.fval, f_u), ...
            utils.errX(x_mosek, u), utils.errX(x_mosek, x_mosek), utils.errX(x_mosek, x_gurobi), ...
            out_mosek.time_cpu, iters_N_mosek, out_mosek.sparsity_x};
        
        table_data{end+1} = {'cvx_gurobi', out_gurobi.fval, utils.errObj(out_gurobi.fval, f_u), ...
            utils.errX(x_gurobi, u), utils.errX(x_gurobi, x_mosek), utils.errX(x_gurobi, x_gurobi), ...
            out_gurobi.time_cpu, iters_N_gurobi, out_gurobi.sparsity_x};
        
        if args.plot
            figure;
            semilogy(out_mosek.iters(:,1), out_mosek.iters(:,2), '.-', 'DisplayName', ...
                ['cvx_mosek in ' num2str(iters_N_mosek) ' iters']);
            hold on;
            semilogy(out_gurobi.iters(:,1), out_gurobi.iters(:,2), '.-', 'DisplayName', ...
                ['cvx_gurobi in ' num2str(iters_N_gurobi) ' iters']);
        end
    end

    % Test each solver
    for i = 1:length(args.solvers)
        solver_name = args.solvers{i};
        if ~ismember(solver_name, utils.solversCollection)
            fprintf('ERROR: %s\n',['Solver ' solver_name ' does not exist, skipping']);
            continue;
        end
        
        if args.compare && ismember(solver_name, {'gl_cvx_gurobi', 'gl_cvx_mosek'})
            fprintf(['Solver ' solver_name ' already tested in compare, skipping']);
            continue;
        end
        
        solver_opts = struct('solver_name', solver_name);
        if isfield(args.opts, solver_name)
            solver_opts = mergestructs(solver_opts, args.opts.(solver_name));
        end
        
        [x, iters_N, out] = utils.testSolver(x0, A, b, mu, solver_opts);
        
        if iters_N <= 0
            fprintf('ERROR: %s\n',[solver_name '''s iters_N = ' num2str(iters_N) ', skipping!']);
            continue;
        end
        
        if args.plot
            semilogy(out.iters(:,1), out.iters(:,2), '.-', 'DisplayName', ...
                [solver_name(4:end) ' in ' num2str(iters_N) ' iters']);
        end
        
        % Add results to table
        if args.compare
            table_data{end+1} = {solver_name(4:end), out.fval, utils.errObj(out.fval, f_u), ...
                utils.errX(x, u), utils.errX(x, x_mosek), utils.errX(x, x_gurobi), ...
                out.time_cpu, iters_N, out.sparsity_x};
        else
            table_data{end+1} = {solver_name(4:end), out.fval, utils.errObj(out.fval, f_u), ...
                utils.errX(x, u), out.time_cpu, iters_N, out.sparsity_x};
        end
    end

    % Display results
    if args.compare
        headers = {'Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'x_CVXmosek_Error', ...
            'x_CVXgurobi_Error', 'Time(s)', 'Iter', 'Sparsity'};
    else
        headers = {'Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'Time(s)', 'Iter', 'Sparsity'};
    end
    
    results_table = cell2table(vertcat(table_data{:}), 'VariableNames', headers);
    disp(results_table);

    if args.plot
        legend('show');
        grid on;
    end

    diary off;
end

function result = mergestructs(s1, s2)
    result = s1;
    fields = fieldnames(s2);
    for i = 1:length(fields)
        result.(fields{i}) = s2.(fields{i});
    end
end

function str = struct2str(s)
    if isempty(s)
        str = 'struct()';
        return;
    end
    
    fields = fieldnames(s);
    str = 'struct(';
    
    for i = 1:length(fields)
        field = fields{i};
        value = s.(field);
        
        % Handle different value types
        if isstruct(value)
            valueStr = struct2str(value);
        elseif ischar(value)
            valueStr = ['''' value ''''];
        elseif isnumeric(value)
            valueStr = mat2str(value);
        elseif islogical(value)
            valueStr = mat2str(value);
        else
            valueStr = 'unknown_type';
        end
        
        % Add field-value pair
        str = [str '''' field ''', ' valueStr];
        
        % Add comma if not last field
        if i < length(fields)
            str = [str, ', '];
        end
    end
    
    str = [str ')'];
end


        