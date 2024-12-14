import numpy as np
import cvxpy as cp
import utilities.utils as utils

def gl_cvx_mosek(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    """
    Solve the group LASSO problem using CVXPY with the MOSEK solver.
    
    Args:
        initial_x (np.ndarray): Initial guess for the variable X.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Additional algorithm options. Defaults to {}.
    
    Returns:
        tuple: Optimal solution X.value, number of iterations iters_N, and output dictionary out.
    """
    # Define the optimization variable with the same shape as the initial guess
    X = cp.Variable(shape=(A.shape[1], b.shape[1]))
    X.value = initial_x

    # Define the objective function: 0.5 * ||A*X - b||_F^2 + mu * sum of L2 norms of the rows of X
    objective = cp.Minimize(
        0.5 * cp.square(cp.norm(A @ X - b, 'fro')) +
        mu * cp.sum(cp.norm(X, p=2, axis=1))
    )

    # Formulate and solve the optimization problem using MOSEK solver
    problem = cp.Problem(objective)
    problem.solve(solver=cp.MOSEK, verbose=True)

    # Read the solver logs from the specified log file
    with open(utils.cvxLogsName, 'r', encoding='utf-8') as log_file:
        logs = log_file.read()

    # Parse the number of iterations from the logs specific to MOSEK
    iterations = utils.parse_iters(logs, 'MOSEK')

    # Logging solver details and results
    utils.logger.debug("#######==Solver: cvx(MOSEK)==#######")
    utils.logger.debug(f"Objective value: {problem.value}")
    utils.logger.debug(f"Status: {problem.status}")
    utils.logger.debug(f"Solver status: {problem.solver_stats}")
    utils.logger.debug(f"#######==CVXPY's Logs:==#######\n{logs}")
    utils.logger.debug("#######==END of Logs:==#######")
    utils.logger.debug(f"Parsed iterations:\n{iterations}")
    
    # Check if any iterations were recorded; if not, log an error
    if len(iterations) == 0:
        utils.logger.error("Solver cvx(MOSEK) recorded zero iterations. Please check stdout redirection!")
        iters_N = -1
    else:
        iters_N = len(iterations)

    # Prepare the output dictionary with iterations and final objective value
    out = {
        'iters': iterations,  # List of tuples containing iteration number and objective value [(iter, fval), ...]
        'fval': problem.value  # Final objective function value
    }

    # Optimal solution, number of iterations, and output dictionary
    return X.value, iters_N, out