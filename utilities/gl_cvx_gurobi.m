function [x, num_iters, out] = gl_cvx_gurobi(initial_x, A, b, mu, opts)
    % Solve the group LASSO problem using CVX with the Gurobi solver.
    %
    % Args:
    %   initial_x (matrix): Initial guess for the variable X
    %   A (matrix): Constraint matrix
    %   b (matrix): Observation vector
    %   mu (double): Regularization parameter
    %   opts (struct): Additional algorithm options (optional)
    %
    % Returns:
    %   x (matrix): Optimal solution
    %   num_iters (int): Number of iterations
    %   out (struct): Output structure with algorithm details
    
    if nargin < 5
        opts = struct();
    end
    
    [~, n] = size(A);
    [~, l] = size(b);
    
    % Use CVX to solve the optimization problem
    cvx_begin
        cvx_solver gurobi
        cvx_quiet false
        variable X(n,l)
        
        % Initialize with initial guess if provided
        X = initial_x;
        
        % Objective function: 0.5 * ||A*X - b||_F^2 + mu * sum of L2 norms of rows of X
        objective = 0.5 * square_pos(norm(A * X - b, 'fro')) + ...
                   mu * sum(norms(X, 2, 2));
        
        minimize(objective)
    cvx_end
    
    % Read solver logs
    fid = fopen(utils.cvxLogsName, 'r');
    logs = fscanf(fid, '%c');
    fclose(fid);
    
    % Parse iterations from Gurobi logs
    iterations = utils.parse_iters(logs, 'GUROBI');
    
    % Print solver details and results
    fprintf('#######==Solver: cvx(GUROBI)==#######\n');
    fprintf('Objective value: %f\n', cvx_optval);
    fprintf('Status: %s\n', cvx_status);
    fprintf('#######==CVXPY''s Logs:==#######\n%s\n', logs);
    fprintf('#######==END of Logs:==#######\n');
    fprintf('Parsed iterations:\n'); disp(iterations);
    
    % Check if iterations were recorded
    if isempty(iterations)
        fprintf('ERROR: Solver cvx(GUROBI) recorded zero iterations. Please check stdout redirection!\n');
        num_iters = -1;
    else
        num_iters = size(iterations, 1);
    end
    
    % Prepare output structure
    out = struct();
    out.iters = iterations;    % Matrix containing iteration number and objective value
    out.fval = cvx_optval;    % Final objective function value
    
    x = X;
end