import numpy as np
import utilities.utils as utils

# Augmented Lagrangian Method (Dual Problem)
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/lect/16-lect-alm-pku.pdf

def updateZ(reference, mu):
    """
    Update the variable Z using the proximal operator.

    Args:
        reference (np.ndarray): Reference matrix for updating Z.
        mu (float): Regularization parameter.

    Returns:
        np.ndarray: Updated Z matrix.
    """
    norm_reference = np.linalg.norm(reference, axis=1, keepdims=True)
    norm_reference[norm_reference < mu] = mu
    updated_z = reference * (mu / norm_reference)
    return updated_z

def gl_ALM_dual(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Solve the dual problem of the group LASSO using the Augmented Lagrangian Method (ALM).

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    # Initialize options with default ALM dual parameters
    opts = utils.ALM_dual_optsInit(opts)
    utils.logger.info(f"optsOuter: \n{opts}")
    
    m, n = A.shape
    _, num_targets = b.shape
    output = utils.outInit()
    output['prim_hist'] = []
    output['dual_hist'] = []
    
    x = initial_x
    sigma = opts['sigma']  # Quadratic penalty coefficient
    inverse_matrix = np.linalg.inv(np.eye(m) + sigma * A @ A.T)
    z = np.zeros((n, num_targets))

    for outer_iter in range(opts['maxit']):
        for inner_iter in range(opts['maxit_inn']):
            # Update dual variable y
            y = inverse_matrix @ (A @ x - sigma * A @ z - b)
            
            # Store previous z for convergence check
            previous_z = z.copy()
            
            # Update primal variable z using the proximal operator
            z = updateZ(x / sigma - A.T @ y, mu)
            
            # Calculate the change in z for convergence
            z_change = np.linalg.norm(z - previous_z, 'fro')
            
            # Increment total inner iterations
            output['itr_inn'] += 1
            
            # Check convergence of the inner loop
            if z_change < opts['thre_inn']:
                break

        # Update primal variable x
        x = x - sigma * (A.T @ y + z)
        
        # Compute the primal objective function value
        primal_obj = utils.objFun(x, A, b, mu)
        
        # Compute the dual objective function value
        dual_obj = 0.5 * np.linalg.norm(y, ord='fro') ** 2 - np.sum(y * b)
        
        # Log the difference between primal and dual objectives
        utils.logger.debug(f"primal_obj - dual_obj: {primal_obj - dual_obj}")
        
        # Record history of objective values
        output['prim_hist'].append(primal_obj)
        output['dual_hist'].append(dual_obj)

        # Check convergence based on the norm of the residual
        if np.linalg.norm(A.T @ y + z) < opts['thre']:
            break
    
    # Store the number of iterations and final objective value
    output['itr'] = outer_iter + 1
    output['fval'] = output['prim_hist'][-1]
    utils.logger.debug(f"ALM_dual: itr: {output['itr']}, fval: {output['fval']}")
    utils.logger.debug(f"ALM_dual: itr_inn: {output['itr_inn']}")
    utils.logger.debug(f"ALM_dual: len(out['prim_hist']): {len(output['prim_hist'])}")
    
    # Prepare iteration history for output
    output['iters'] = zip(range(output['itr']), output['prim_hist'])  #, output['dual_hist'])
        
    return x, output['itr'], output