from mosek.fusion import *
import sys
import utilities.utils as utils
import numpy as np

def gl_mosek(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    """
    Solve the group LASSO problem using MOSEK through its Python interface.
    
    Args:
        initial_x (np.ndarray): Initial guess for the variable X.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Additional algorithm options. Defaults to {}.
    
    Returns:
        tuple: Optimal solution X.level().reshape(n, l), number of iterations iters_N, and output dictionary out.
    """
    utils.logger.debug('Note: for MOSEK, initial_x is ignored')
    num_constraints, num_features = A.shape
    num_targets = b.shape[1]
    
    # Initialize MOSEK model
    with Model('gl_mosek') as model:
        model.setLogHandler(sys.stdout)
        
        # Convert numpy arrays to MOSEK matrices
        constraint_matrix = Matrix.dense(A)
        observation_vector = Matrix.dense(b)
        
        # Define optimization variables
        X = model.variable("X", [num_features, num_targets], Domain.unbounded())
        X.setLevel(initial_x)
        Y = model.variable("Y", [num_constraints, num_targets], Domain.unbounded())
        t_scalar = model.variable("t1", 1)
        t_variables = model.variable("t_variables", num_features, Domain.greaterThan(0.0))
        
        # Add constraints: A * X[:, j] - b[:, j] == Y[:, j] for each target j
        for j in range(num_targets):
            model.constraint(Expr.sub(Expr.sub(Expr.mul(constraint_matrix, X.slice([0, j], [num_constraints, j+1])), observation_vector), Y.slice([0, j], [num_constraints, j+1])), Domain.equalsTo(0.0), f"Constraint_Y_{j}")
        
        # Add quadratic cone constraints
        model.constraint(Expr.vstack([Expr.add(1.0, t_scalar), Expr.mul(2.0, Y.reshape(num_constraints * num_targets)), Expr.sub(1.0, t_scalar)]), Domain.inQCone(), "QCone_Constraint")
        
        for i in range(num_features):
            model.constraint(
                Expr.vstack([
                    t_variables.index(i),
                    X.slice([i, 0], [i + 1, num_targets]).reshape(num_targets)
                ]),
                Domain.inQCone(),
                f"QCone_Feature_{i}"
            )
        
        # Define the objective function: 0.5 * t1 + mu * sum(t_variables)
        objective_expression = Expr.add(Expr.mul(0.5, t_scalar), Expr.mul(mu, Expr.sum(t_variables)))
        model.objective('Objective', ObjectiveSense.Minimize, objective_expression)
        
        # Solve the optimization problem
        model.solve()
        
        # Read the solver logs from the specified log file
        with open(utils.cvxLogsName, encoding='utf-8') as log_file:
            logs = log_file.read()
        
        # Parse the number of iterations from the logs specific to MOSEK
        iterations = utils.parse_iters(logs, 'MOSEK_OLD')
        
        # Logging solver details and results
        utils.logger.debug("#######==Solver: MOSEK==#######")
        utils.logger.debug(f"Objective value: {model.primalObjValue()}")
        utils.logger.debug(f"#######==MOSEK's Logs:==#######\n{logs}")
        utils.logger.debug(f"#######==END of Logs:==#######")
        utils.logger.debug(f"Iterations after parsing:\n{iterations}")
        
        # Check if any iterations were recorded; if not, log an error
        if len(iterations) == 0:
            utils.logger.error("Solver MOSEK recorded zero iterations. Please check stdout redirection!")
            iters_N = -1
        else:
            iters_N = len(iterations)
        
        # Prepare the output dictionary with iterations and final objective value
        output = {
            'iters': iterations,  # List of tuples containing iteration number and objective value [(iter, fval), ...]
            'fval': model.primalObjValue()  # Final objective function value
        }
        
        # Retrieve and reshape the optimal solution
        optimal_X = X.level().reshape(num_features, num_targets)
        
        return optimal_X, iters_N, output