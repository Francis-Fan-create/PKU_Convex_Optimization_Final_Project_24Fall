function [x, num_iters, out] = gl_mosek(initial_x, A, b, mu, opts)
    % Solve the group LASSO problem using MOSEK through its MATLAB interface.
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
    
    fprintf('Note: for MOSEK, initial_x is ignored\n');
    [num_constraints, num_features] = size(A);
    num_targets = size(b, 2);
    
    % Initialize MOSEK model
    model = mosekopt.Model('gl_mosek');
    
    % Define optimization variables
    X = model.variable('X', [num_features, num_targets], Domain.unbounded());
    X.setLevel(initial_x);
    Y = model.variable('Y', [num_constraints, num_targets], Domain.unbounded());
    t_scalar = model.variable('t1', 1);
    t_variables = model.variable('t_variables', num_features, Domain.greaterThan(0.0));
    
    % Add constraints: A * X[:, j] - b[:, j] == Y[:, j] for each target j
    for j = 1:num_targets
        model.constraint(...
            Expr.sub(...
                Expr.sub(...
                    Expr.mul(Matrix.dense(A), ...
                    X.slice([1, j], [num_constraints, j+1])), ...
                    Matrix.dense(b)), ...
                Y.slice([1, j], [num_constraints, j+1])), ...
            Domain.equalsTo(0.0), ...
            sprintf('Constraint_Y_%d', j));
    end
    
    % Add quadratic cone constraints
    model.constraint(...
        Expr.vstack([...
            Expr.add(1.0, t_scalar), ...
            Expr.mul(2.0, Y.reshape(num_constraints * num_targets)), ...
            Expr.sub(1.0, t_scalar)...
        ]), ...
        Domain.inQCone(), 'QCone_Constraint');
    
    for i = 1:num_features
        model.constraint(...
            Expr.vstack([...
                t_variables.index(i), ...
                X.slice([i, 1], [i + 1, num_targets]).reshape(num_targets)...
            ]), ...
            Domain.inQCone(), ...
            sprintf('QCone_Feature_%d', i));
    end
    
    % Define objective function: 0.5 * t1 + mu * sum(t_variables)
    objective_expression = Expr.add(...
        Expr.mul(0.5, t_scalar), ...
        Expr.mul(mu, Expr.sum(t_variables)));
    model.objective('Objective', ObjectiveSense.Minimize, objective_expression);
    
    % Solve the optimization problem
    model.solve();
    
    % Read solver logs
    fid = fopen(utils.cvxLogsName, 'r');
    logs = fscanf(fid, '%c');
    fclose(fid);
    
    % Parse iterations from MOSEK logs
    iterations = utils.parse_iters(logs, 'MOSEK_OLD');
    
    % Print solver details and results
    fprintf('#######==Solver: MOSEK==#######\n');
    fprintf('Objective value: %f\n', model.primalObjValue());
    fprintf('#######==MOSEK''s Logs:==#######\n%s\n', logs);
    fprintf('#######==END of Logs:==#######\n');
    fprintf('Parsed iterations:\n'); disp(iterations);
    
    % Check if iterations were recorded
    if isempty(iterations)
        fprintf('ERROR: Solver MOSEK recorded zero iterations. Please check stdout redirection!\n');
        num_iters = -1;
    else
        num_iters = size(iterations, 1);
    end
    
    % Prepare output structure
    out = struct();
    out.iters = iterations;
    out.fval = model.primalObjValue();
    
    % Extract solution
    x = reshape(X.level(), num_features, num_targets);
end