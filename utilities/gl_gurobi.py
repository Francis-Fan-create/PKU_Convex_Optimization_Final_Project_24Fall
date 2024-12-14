import gurobipy as gp
import numpy as np
import utilities.utils as utils

def gl_gurobi(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    """
    Solve the group LASSO problem using Gurobi through its Python interface.
    
    Args:
        initial_x (np.ndarray): Initial guess for the variable X.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Additional algorithm options. Defaults to {}.
    
    Returns:
        tuple: Optimal solution X.x, number of iterations iters_N, and output dictionary out.
    """
    m, n = A.shape
    num_targets = b.shape[1]
    
    # Initialize Gurobi model
    model = gp.Model()
    
    # Add optimization variables
    X = model.addMVar((n, num_targets), lb=-gp.GRB.INFINITY, name="X")
    X.start = initial_x
    Y = model.addMVar((m, num_targets), lb=-gp.GRB.INFINITY, name="Y")
    t_variables = model.addMVar(n, lb=0.0, name="t")
    
    # Add constraints: A * X[:, j] - b[:, j] == Y[:, j] for each target j
    for j in range(num_targets):
        model.addConstr(A @ X[:, j] - b[:, j] == Y[:, j], name=f"Constraint_Y_{j}")
    
    # Add constraints: ||X[i, :]||_2 <= t[i] for each feature i
    for i in range(n):
        model.addConstr(gp.quicksum(X[i, k] * X[i, k] for k in range(num_targets)) <= t_variables[i] * t_variables[i],
                        name=f"Constraint_t_{i}")
    
    # Define the objective function: 0.5 * ||A*X - b||_F^2 + mu * sum(t)
    objective = 0.5 * gp.quicksum(Y[:, j].dot(Y[:, j]) for j in range(num_targets)) + mu * gp.quicksum(t_variables[i] for i in range(n))
    model.setObjective(objective, gp.GRB.MINIMIZE)
    
    # Optimize the model
    model.optimize()
    
    # Read the solver logs from the specified log file
    with open(utils.cvxLogsName, 'r', encoding='utf-8') as log_file:
        logs = log_file.read()
    
    # Parse the number of iterations from the logs specific to Gurobi
    iterations = utils.parse_iters(logs, 'GUROBI')
    
    # Logging solver details and results
    utils.logger.debug("#######==Solver: GUROBIPY==#######")
    utils.logger.debug(f"Objective value: {model.objVal}")
    utils.logger.debug(f"Status: {model.status}")
    utils.logger.debug(f"#######==Gurobi's Logs:==#######\n{logs}")
    utils.logger.debug(f"#######==END of Logs:==#######")
    utils.logger.debug(f"Parsed iterations:\n{iterations}")
    
    # Check if any iterations were recorded; if not, log an error
    if len(iterations) == 0:
        utils.logger.error("Solver GUROBI recorded zero iterations. Please check stdout redirection!")
        iters_N = -1
    else:
        iters_N = len(iterations)
    
    # Prepare the output dictionary with iterations and final objective value
    out = {
        'iters': iterations,          # List of tuples containing iteration number and objective value [(iter, fval), ...]
        'fval': model.objVal          # Final objective function value
    }
    
    # Return the optimal solution, number of iterations, and output dictionary
    return X.x, iters_N, out