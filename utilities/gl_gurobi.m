function [x, num_iters, out] = gl_gurobi(initial_x, A, b, mu, opts)
    % Solve the group LASSO problem using Gurobi through its MATLAB interface.
    %
    % Args:
    %   initial_x (matrix): Initial guess for the variable X
    %   A (matrix): Constraint matrix
    %   b (matrix): Observation vector
    %   mu (double): Regularization parameter
    %   opts (struct): Additional algorithm options (optional)
    
    if nargin < 5
        opts = struct();
    end
    
    [m, n] = size(A);
    num_targets = size(b, 2);
    
    % Initialize Gurobi model
    model = struct();
    model.modelsense = 'min';
    
    % Setup variables
    % X variables: n x num_targets matrix
    num_vars = n * num_targets + m * num_targets + n;
    model.lb = [-inf(n * num_targets, 1); -inf(m * num_targets, 1); zeros(n, 1)];
    model.ub = [inf(n * num_targets, 1); inf(m * num_targets, 1); inf(n, 1)];
    model.start = [reshape(initial_x, [], 1); zeros(m * num_targets + n, 1)];
    
    % Setup quadratic objective
    % Objective: 0.5 * ||Y||_F^2 + mu * sum(t)
    Q = sparse(num_vars, num_vars);
    Q(n*num_targets+1:n*num_targets+m*num_targets, n*num_targets+1:n*num_targets+m*num_targets) = 0.5 * speye(m*num_targets);
    model.Q = Q;
    c = zeros(num_vars, 1);
    c(end-n+1:end) = mu;
    model.obj = c;
    
    % Setup constraints
    % A * X[:, j] - b[:, j] == Y[:, j] for each target j
    Aeq = sparse(m * num_targets, num_vars);
    beq = reshape(b, [], 1);
    
    for j = 1:num_targets
        rows = (j-1)*m + (1:m);
        cols_x = (j-1)*n + (1:n);
        cols_y = n*num_targets + (j-1)*m + (1:m);
        Aeq(rows, cols_x) = A;
        Aeq(rows, cols_y) = -speye(m);
    end
    model.A = [Aeq; sparse(n, num_vars)];
    model.rhs = [beq; zeros(n, 1)];
    model.sense = [repmat('=', m*num_targets, 1); repmat('<', n, 1)];
    
    % Add SOC constraints: ||X[i, :]||_2 <= t[i]
    model.quadcon = struct('Qc', cell(1,n), 'q', cell(1,n), 'rhs', cell(1,n));
    for i = 1:n
        Qc = sparse(num_vars, num_vars);
        for k = 1:num_targets
            idx = (k-1)*n + i;
            Qc(idx, idx) = 1;
        end
        q = zeros(num_vars, 1);
        q(end-n+i) = -1;
        model.quadcon(i).Qc = Qc;
        model.quadcon(i).q = q;
        model.quadcon(i).rhs = 0;
    end
    
    % Set Gurobi parameters
    params.OutputFlag = 1;
    params.LogFile = utils.cvxLogsName;
    
    % Solve the model
    result = gurobi(model, params);
    
    % Read solver logs
    fid = fopen(utils.cvxLogsName, 'r');
    logs = fscanf(fid, '%c');
    fclose(fid);
    
    % Parse iterations from Gurobi logs
    iterations = utils.parse_iters(logs, 'GUROBI');
    
    % Print solver details and results
    fprintf('#######==Solver: GUROBI==#######\n');
    fprintf('Objective value: %f\n', result.objval);
    fprintf('Status: %s\n', result.status);
    fprintf('#######==Gurobi''s Logs:==#######\n%s\n', logs);
    fprintf('#######==END of Logs:==#######\n');
    fprintf('Parsed iterations:\n'); disp(iterations);
    
    % Check if iterations were recorded
    if isempty(iterations)
        fprintf('ERROR: Solver GUROBI recorded zero iterations. Please check stdout redirection!\n');
        num_iters = -1;
    else
        num_iters = size(iterations, 1);
    end
    
    % Prepare output structure
    out = struct();
    out.iters = iterations;
    out.fval = result.objval;
    
    % Extract solution
    x = reshape(result.x(1:n*num_targets), n, num_targets);
end