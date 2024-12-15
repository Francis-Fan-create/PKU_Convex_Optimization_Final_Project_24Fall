import numpy as np
import utilities.utils as utils

# Alternating Direction Method of Multipliers (Dual Problem)
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/lect/24-lect-admm-chhyx.pdf
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_admm/LASSO_admm_primal.html

def updateZ(reference, mu):
    """
    Update the variable Z using the proximal operator.

    Args:
        reference (np.ndarray): Reference matrix for updating Z.
        mu (float): Regularization parameter.

    Returns:
        np.ndarray: Updated Z matrix.
    """
    norm_ref = np.linalg.norm(reference, axis=1, keepdims=True)
    norm_ref[norm_ref < mu] = mu
    updated_z = reference * (mu / norm_ref)
    return updated_z


def gl_ADMM_dual(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Solve the dual problem of the group LASSO using the ADMM algorithm.

    Args:
        initial_x (np.ndarray): Initial guess for the primal variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    # Initialize options with default ADMM dual parameters
    opts = utils.ADMM_dual_optsInit(opts)
    utils.logger.info(f"optsOuter: \n{opts}")
    
    m, n = A.shape
    _, l = b.shape
    output = utils.outInit()
    output['prim_hist'] = []
    output['dual_hist'] = []

    x = initial_x
    sigma = opts['sigma']   # Quadratic penalty coefficient
    inverse_matrix = np.linalg.inv(np.eye(m) + sigma * A @ A.T)
    z = np.zeros((n, l))

    for iteration in range(opts['maxit']):
        # Update dual variable y
        y = inverse_matrix @ (A @ x - sigma * A @ z - b)
        
        # Update primal variable z using the proximal operator
        z = updateZ(x / sigma - A.T @ y, mu)
        
        # Update primal variable x
        x = x - sigma * (A.T @ y + z)

        # Compute the primal objective function value
        primal_obj = utils.objFun(x, A, b, mu)
        
        # Compute the dual objective function value
        dual_obj = 0.5 * np.linalg.norm(y, ord='fro') ** 2 + np.sum(y * b)
        
        # Record history of objective values
        output['prim_hist'].append(primal_obj)
        output['dual_hist'].append(dual_obj)

        # Check convergence based on the norm of the residual
        if np.linalg.norm(A.T @ y + z) < opts['thre']:
            break
    
    # Store the number of iterations and final objective value
    output['itr'] = iteration + 1
    output['fval'] = output['prim_hist'][-1]
    utils.logger.debug(f"ADMM_dual: itr: {output['itr']}, fval: {output['fval']}")
    utils.logger.debug(f"ADMM_dual: len(out['prim_hist']): {len(output['prim_hist'])}")
    
    # Prepare iteration history for output
    output['iters'] = zip(range(output['itr']), output['prim_hist'])  #, output['dual_hist'])
    
    return x, output['itr'], output