import numpy as np
import utilities.utils as utils

# Fast Proximal Gradient Method
# Reference: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_proxg/demo_proxg.html

def gl_ProxGD_primal_inner(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Inner function for the Fast Proximal Gradient Method to solve the group LASSO problem.

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    opts = utils.optsInnerInit(opts)
    utils.logger.debug(f"--->optsInner:<--- \n{opts}")
    
    target_mu = opts['mu0']  # Target minimum mu0. Using continuation strategy, current mu >= mu0
    step_size = opts['alpha0']  # Initial step size

    # Initial computation
    x = initial_x
    y = x
    previous_x = initial_x
    gradient = np.matmul(A.T, np.matmul(A, y) - b)
    temp = 0.5 * np.linalg.norm(np.matmul(A, y) - b, ord='fro') ** 2
    current_val = temp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f_val = temp + target_mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    gradient_norm = np.linalg.norm(x - utils.prox(x - gradient, mu), ord="fro")
    
    output = utils.outInit()
    best_f_val = 1e7

    for iteration in np.arange(opts['maxit_inn']):
        previous_y = y
        previous_gradient = gradient
        previous_x = x

        output['g_hist'].append(gradient_norm)
        output['f_hist_inner'].append(f_val)
        best_f_val = min(best_f_val, f_val)

        output['f_hist_best'].append(best_f_val)

        utils.logger.debug(f"  inner iter {iteration}: fval: {f_val}, f_best: {best_f_val}")
        utils.logger.debug(f"\tabs(fval - f_best) = {np.abs(f_val - best_f_val)}")
        utils.logger.debug(f"\topts['ftol'] = {opts['ftol']}")
        utils.logger.debug(f"\tout['f_hist_inner'][{iteration}] - out['f_hist_inner'][{iteration - 1}] = {output['f_hist_inner'][iteration] - output['f_hist_inner'][iteration - 1]}")
        utils.logger.debug(f"\topts['gtol'] = {opts['gtol']}")
        utils.logger.debug(f"\tout['g_hist'][{iteration}] = {output['g_hist'][iteration]}")

        if iteration > 2 and np.abs(output['f_hist_inner'][iteration] - output['f_hist_inner'][iteration - 1]) < opts['ftol'] and output['g_hist'][iteration] < opts['gtol']:
            output['flag'] = True
            break

        # Line search
        for nls_attempt in np.arange(10):
            x_candidate = utils.prox(previous_x - step_size * gradient, step_size * mu)
            temp_candidate = 0.5 * np.linalg.norm(np.matmul(A, x_candidate) - b, ord='fro') ** 2
            f_candidate = temp_candidate + mu * np.sum(np.linalg.norm(x_candidate, ord=2, axis=1))

            if f_candidate <= current_val - 0.5 * step_size * opts['rhols'] * gradient_norm ** 2:
                x = x_candidate
                break
            step_size *= opts['eta']

        theta = (iteration - 1) / (iteration + 2)  # Iteration starts at 0
        y = x + theta * (x - previous_x)
        residual = np.matmul(A, y) - b
        gradient = np.matmul(A.T, residual)
        f_val = temp_candidate + target_mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

        # Barzilai-Borwein (BB) step size update
        step_size = utils.BBupdate(y, previous_y, gradient, previous_gradient, iteration, step_size)

        gradient_norm = np.linalg.norm(x - y, ord='fro') / step_size
        previous_Q = opts['Q']
        opts['Q'] = opts['gamma'] * previous_Q + 1
        current_val = (opts['gamma'] * previous_Q * current_val + f_candidate) / opts['Q']

    output['itr'] = iteration + 1
    output['flag'] = True
    return x, output['itr'], output


def gl_ProxGD_primal(initial_x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: dict = {}):
    """
    Solve the group LASSO problem using the Proximal Gradient Method.

    Args:
        initial_x (np.ndarray): Initial guess for the variable x.
        A (np.ndarray): Constraint matrix.
        b (np.ndarray): Observation vector.
        mu (float): Regularization parameter.
        opts (dict, optional): Algorithm options. Defaults to {}.

    Returns:
        tuple: Updated x, number of iterations, and output dictionary.
    """
    opts = utils.ProxGD_primal_optsInit(opts)
    opts['method'] = gl_ProxGD_primal_inner
    x, iterations, output = utils.LASSO_group_con(initial_x, A, b, mu, opts)
    return x, iterations, output