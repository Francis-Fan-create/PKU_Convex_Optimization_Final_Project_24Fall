import numpy as np
import utilities.utils as utils

# Alternating Direction Method of Multipliers (Primal Problem)
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_admm/demo_admm.html
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_admm/LASSO_admm_primal.html

def gl_ADMM_primal(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Solve the primal problem of the group LASSO using the ADMM algorithm.

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    # Initialize options with default ADMM primal parameters
    opts = utils.ADMM_primal_optsInit(opts)
    _, num_features = A.shape
    _, num_targets = b.shape
    output = utils.outInit()
    output['prim_hist'] = []

    x = initial_x
    y = initial_x
    z = np.zeros((num_features, num_targets))

    sigma = opts['sigma']   # Quadratic penalty coefficient
    inverse_matrix = np.linalg.inv(sigma * np.eye(num_features) + A.T @ A) 
    # Since the penalty factor does not change during iterations, precomputing the inverse can accelerate the process.
    A_transpose_b = A.T @ b

    for iteration in range(opts['maxit']):
        # Update x
        x = inverse_matrix @ (sigma * y + A_transpose_b - z)
        
        # Store previous y for convergence check
        previous_y = y.copy()
        
        # Update y using the proximal operator
        y = utils.prox(x + z / sigma, mu / sigma)
        
        # Update dual variable z
        z = z + sigma * (x - y)

        # Calculate primal and dual residuals for convergence
        primal_residual = np.linalg.norm(x - y)
        dual_residual = np.linalg.norm(previous_y - y)
        
        # Compute the primal objective function value
        objective_value = utils.objFun(x, A, b, mu)
        output['prim_hist'].append(objective_value)

        # Check for convergence
        if primal_residual < opts['thre'] and dual_residual < opts['thre']:
            break

    # Store the number of iterations and final objective value
    output['itr'] = iteration + 1
    output['fval'] = objective_value
    utils.logger.info(f"ADMM_primal: itr: {output['itr']}, fval: {output['fval']}")
    utils.logger.info(f"ADMM_primal: len(out['prim_hist']): {len(output['prim_hist'])}")
    
    # Prepare iteration history for output
    output['iters'] = zip(range(output['itr']), output['prim_hist'])

    return x, output['itr'], output