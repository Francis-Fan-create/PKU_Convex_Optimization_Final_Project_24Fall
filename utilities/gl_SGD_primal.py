import numpy as np
import utilities.utils as utils

# Subgradient Method for Solving the Primal Problem (Using Continuation Strategy)
# References:
# http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_subgrad/LASSO_subgrad_inn.html
# http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_subgrad/demo_cont.html

def gl_SGD_primal_inner(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Inner function for the Subgradient Method to solve the group LASSO problem.

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    options = utils.optsInnerInit(opts)
    utils.logger.debug(f"--->optsInner:<--- \n{options}")

    target_mu = options['mu0']  # Target minimum mu0. Using continuation strategy, current mu >= mu0
    step_size = options['alpha0']  # Initial step size

    # Initial computation
    current_x = initial_x
    residual = np.matmul(A, current_x) - b
    gradient = np.matmul(A.T, residual)
    norm_x = np.linalg.norm(current_x, axis=1).reshape((-1, 1))
    subgradient = current_x / ((norm_x <= 1e-6) + norm_x)
    subgradient = subgradient * mu + gradient
    gradient_norm = np.linalg.norm(gradient, ord="fro")
    temp_value = 0.5 * np.linalg.norm(residual, ord='fro') ** 2
    current_val = temp_value + mu * np.sum(np.linalg.norm(current_x, ord=2, axis=1))
    function_val = temp_value + target_mu * np.sum(np.linalg.norm(current_x, ord=2, axis=1))
    utils.logger.debug(f"inner function value: {function_val}")

    output = utils.outInit()
    # best_f_val = 1e7  # Uncomment if tracking best function value

    for iteration in np.arange(options['maxit_inn']):
        previous_gradient = gradient
        previous_x = current_x

        output['g_hist'].append(gradient_norm)
        output['f_hist_inner'].append(function_val)

        # Check for convergence
        if iteration > 2 and np.abs(output['f_hist_inner'][iteration] - output['f_hist_inner'][iteration - 1]) < options['ftol']:
            output['flag'] = True
            break

        # Line search
        for nls_attempt in np.arange(10):
            candidate_x = utils.prox(previous_x - step_size * subgradient, step_size * mu)
            candidate_residual = 0.5 * np.linalg.norm(np.matmul(A, candidate_x) - b, ord='fro') ** 2
            candidate_f = candidate_residual + mu * np.sum(np.linalg.norm(candidate_x, ord=2, axis=1))

            if candidate_f <= current_val - 0.5 * step_size * options['rhols'] * gradient_norm ** 2:
                current_x = candidate_x
                break
            step_size *= options['eta']

        # Update gradients after x update
        gradient = np.matmul(A.T, np.matmul(A, current_x) - b)
        norm_x = np.linalg.norm(current_x, axis=1).reshape((-1, 1))
        subgradient = current_x / ((norm_x <= 1e-6) + norm_x)
        subgradient = subgradient * mu + gradient

        # Update function value
        function_val = candidate_residual + target_mu * np.sum(np.linalg.norm(current_x, ord=2, axis=1))

        # Update gradient norm
        gradient_norm = np.linalg.norm(gradient, ord='fro')

        previous_Q = options['Q']
        options['Q'] = options['gamma'] * previous_Q + 1
        options['Cval'] = (options['gamma'] * previous_Q * current_val + candidate_f) / options['Q']

        # Barzilai-Borwein (BB) step size update
        step_size = utils.BBupdate(current_x, previous_x, gradient, previous_gradient, iteration, step_size)

    output['itr'] = iteration + 1
    return current_x, output['itr'], output

def gl_SGD_primal(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts={}):
    """
    Subgradient method for the primal problem.
    - Minimize (1/2)*||Ax - b||_2^2 + mu*||x||_{1,2}
    - A is m x n, b is m x l, x is n x l

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    options = utils.SGD_primal_optsInit(opts)
    options['method'] = gl_SGD_primal_inner
    x, iterations, out = utils.LASSO_group_con(initial_x, A, b, mu, options)
    return x, iterations, out
    
    