import sys
import os
import re
import time
import logging
import importlib
import numpy as np

def setLoggerLevel(logger, level: str):
    """Set the logging level.

    Args:
        - level (str): Logging level, options are DEBUG, INFO, WARNING, ERROR, CRITICAL. Defaults to INFO.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid logging level input: %s' % level)
    logger.setLevel(numeric_level)
    logger.info(f"Logging level set to {level.upper()}")

# Initialize logger object
def loggerInit(name: str = None):
    # Use a logger with the specified name
    logger = logging.getLogger(name)
    # Set logger level to INFO
    setLoggerLevel(logger, 'INFO')

    # Create a StreamHandler to output logs to console
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(handler)

    # Also write logs to file
    current_work_dir = os.path.dirname(__file__)
    if not os.path.exists(current_work_dir + '/logs'):
        os.makedirs(current_work_dir + '/logs')
    now = time.strftime("%m%d-%H%M%S", time.localtime(time.time()))
    loggerName = current_work_dir + f'/logs/{name}-{now}.log'
    cvxLogsName = current_work_dir + f'/logs/gl_cvx.log'
    logging.basicConfig(
        filename=loggerName,
        level=logging.INFO,
        encoding='utf-8',
        format='[%(asctime)s] %(filename)s: %(funcName)s: %(levelname)s: %(message)s'
    )
    logger.debug(f"Log file saved at: {current_work_dir}\\logs\\{name}{now}.log")
    return logger, loggerName, cvxLogsName

logger, loggerName, cvxLogsName = loggerInit('AGLP')

# Redirect stdout
class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

re_iterc_default = re.compile(r'^ *(?P<iterc>\d{1,3})\:? +(?P<objv>[0-9\.eE\+\-]+)', re.MULTILINE)

# Dictionary of regular expressions
reg_solver = {
    'GUROBI': re_iterc_default,
    # Regular expression to parse MOSEK output
    'MOSEK': re.compile(
        r'^ *([\s\S]{26})\:( +)(?P<iterc>\d{1,2}) ([\s\S]{38})( +)(?P<objv>[\-\+0-9\.eE]+)',
        re.MULTILINE
    ),
    'MOSEK_OLD': re.compile(
        r'^ *(?P<iterc>\d{1,3})\:?( +(?:[0-9\.eE\+\-]+)){4} +(?P<objv>[0-9\.eE\+\-]+)',
        re.MULTILINE
    ),  # Skip four columns
    'CVXOPT': re_iterc_default,
}

# Parse CVX output log file
def parse_iters(log_str, solver=None):
    re_iterc = reg_solver[solver] if solver in reg_solver else re_iterc_default
    results = []
    for match in re_iterc.finditer(log_str):
        results.append((
            int(match.groupdict()['iterc']),
            float(match.groupdict()['objv'])
        ))
    return results

# Clean up CVX output log file
def cleanUpLog():
    with open(cvxLogsName, 'w+', encoding='utf-8') as log_file:
        log_file.truncate(0)
        for line in log_file.readlines():
            line.replace(r'\0', '')

# Calculate sparsity
def sparsity(x) -> float:
    return np.sum(np.abs(x) > 1e-5) / x.size

# Calculate difference between solutions
def errX(x, x0) -> float:
    return np.linalg.norm(x - x0, 'fro') / (1 + np.linalg.norm(x0, 'fro'))

# Calculate objective function value
def objFun(x, A, b, mu) -> float:
    return 0.5 * np.linalg.norm(A @ x - b, 'fro') ** 2 + mu * np.sum(np.linalg.norm(x, axis=1))

# Calculate absolute difference between objective function values
def errObj(obj, obj0) -> float:
    return np.abs(obj - obj0)

# Proximal operator
def prox(x, mu):
    norm_x = np.linalg.norm(x, axis=1, keepdims=True)
    mask = norm_x > mu
    prox_x = x - mu * x / (norm_x + 1e-10)
    prox_x = prox_x * mask
    return prox_x

# BB step size update
def BBupdate(x, x_prev, grad, grad_prev, k, alpha):
    dx = x - x_prev
    dg = grad - grad_prev
    dxg = np.abs(np.sum(dx * dg))
    if dxg > 1e-12:
        if k % 2 == 1:
            alpha = (np.sum(dx * dx) / dxg)
        else:
            alpha = (dxg / np.sum(dg * dg))
    return max(min(alpha, 1e12), 1e-12)

def testDataParams(opts0: dict = {}):
    opts = {}
    opts['seed'] = int(opts0.get("seed", 97108120))  # seed = ord("a") ord("l") ord("x")
    opts['mu'] = float(opts0.get("mu", 1e-2))
    opts['n'] = int(opts0.get("n", 512))
    opts['m'] = int(opts0.get("m", 256))
    opts['l'] = int(opts0.get("l", 2))
    opts['r'] = float(opts0.get("r", 1e-1))
    return opts

# Generate test data
def testData(opts: dict = {}):
    opts = testDataParams(opts)
    logger.info(f"testData opts: {opts}")
    np.random.seed(opts['seed'])
    A = np.random.randn(opts['m'], opts['n'])
    k = round(opts['n'] * opts['r'])
    indices = np.random.permutation(opts['n'])[:k]
    u = np.zeros((opts['n'], opts['l']))
    u[indices, :] = np.random.randn(k, opts['l'])
    b = A @ u
    x0 = np.random.randn(opts['n'], opts['l'])
    f_u = objFun(u, A, b, opts['mu'])
    return x0, A, b, opts['mu'], u, f_u

def testSolver(x0, A, b, mu, opts: dict = {}):
    # Get solver name
    solver_name = opts.get('solver_name', '')
    if solver_name == '':
        raise ValueError('The opts dictionary must contain the key "solver_name" to specify the solver name')
    # Check if solver exists
    try:
        solver = getattr(importlib.import_module("src." + solver_name), solver_name)
    except AttributeError:
        logger.error(f"Solver {solver_name} does not exist, skipping this solver.")
        return None, None, None
    logger.info(f"\n--->Current Test Solver: {solver_name}<---")
    # Extract solver parameters
    solver_opts = dict(opts.get(solver_name[3:], {}))
    logger.info(f"solver_opts: {solver_opts}")
    # Test solver and record time
    start_time = time.time()
    x, num_iters, out = solver(x0, A, b, mu, solver_opts)
    end_time = time.time()
    time_cpu = end_time - start_time
    cleanUpLog()
    out['time_cpu'] = time_cpu
    sparsity_x = sparsity(x)
    out['sparsity_x'] = sparsity_x
    logger.info(f"{solver_name[3:]} takes {time_cpu:.5f}s, with {num_iters} iterations")
    logger.debug(f"out['fval']: {out['fval']}")
    logger.debug(f"sparsity_x: {sparsity_x}")
    logger.debug(f"out['iters']: \n{out['iters']}")
    return x, num_iters, out

# All solvers implemented in the project
solversCollection = [
    'gl_cvx_mosek',
    'gl_cvx_gurobi',
    'gl_mosek',
    'gl_gurobi',
    'gl_SGD_primal',
    'gl_ProxGD_primal',
    'gl_FProxGD_primal',
    'gl_ALM_dual',
    'gl_ADMM_dual',
    'gl_ADMM_primal'
]

# Default parameters for SGD primal method
def SGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50))  # Maximum iterations for the continuation strategy
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250))  # Maximum iterations for the inner loop

    opts['ftol'] = float(opts0.get('ftol', 1e-9))  # Stopping criterion for function value
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6))  # Initial ratio for ftol
    opts['etaf'] = float(opts0.get('etaf', 0.1))  # Reduction factor for ftol in each outer iteration

    opts['gtol'] = float(opts0.get('gtol', 1e-6))  # Stopping criterion for gradient norm
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol']))  # Initial ratio for gtol
    opts['etag'] = float(opts0.get('etag', 0.1))  # Reduction factor for gtol in each outer iteration

    opts['factor'] = float(opts0.get('factor', 0.1))  # Decay rate for regularization parameter
    opts['mu1'] = float(opts0.get('mu1', 10))  # Initial regularization parameter for continuation strategy

    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False))  # Whether to print only outer loop info
    opts['method'] = opts0.get('method', None)  # Solver used in inner loop

    # Parameters for inner loop
    opts['gamma'] = float(opts0.get('gamma', 0.9))
    opts['rhols'] = float(opts0.get('rhols', 1e-6))  # Line search parameter
    opts['eta'] = float(opts0.get('eta', 0.2))  # Line search parameter
    opts['Q'] = float(opts0.get('Q', 1))  # Line search parameter
    return opts

# Default parameters for Proximal Gradient Descent method
def ProxGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50))
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250))

    opts['ftol'] = float(opts0.get('ftol', 1e-9))
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6))
    opts['etaf'] = float(opts0.get('etaf', 0.1))

    opts['gtol'] = float(opts0.get('gtol', 1e-6))
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol']))
    opts['etag'] = float(opts0.get('etag', 0.1))

    opts['factor'] = float(opts0.get('factor', 0.1))
    opts['mu1'] = float(opts0.get('mu1', 10))

    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False))
    opts['method'] = opts0.get('method', None)

    opts['gamma'] = float(opts0.get('gamma', 0.85))
    opts['rhols'] = float(opts0.get('rhols', 1e-6))
    opts['eta'] = float(opts0.get('eta', 0.2))
    opts['Q'] = float(opts0.get('Q', 1))

    return opts

# Default parameters for Fast Proximal Gradient Descent method
def FProxGD_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['maxit'] = int(opts0.get('maxit', 50))
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 250))

    opts['ftol'] = float(opts0.get('ftol', 1e-9))
    opts['ftol_init_ratio'] = float(opts0.get('ftol_init_ratio', 1e6))
    opts['etaf'] = float(opts0.get('etaf', 0.1))

    opts['gtol'] = float(opts0.get('gtol', 1e-6))
    opts['gtol_init_ratio'] = float(opts0.get('gtol_init_ratio', 1 / opts['gtol']))
    opts['etag'] = float(opts0.get('etag', 0.1))

    opts['factor'] = float(opts0.get('factor', 0.1))
    opts['mu1'] = float(opts0.get('mu1', 10))

    opts['is_only_print_outer'] = bool(opts0.get('is_only_print_outer', False))
    opts['method'] = opts0.get('method', None)

    opts['gamma'] = float(opts0.get('gamma', 0.85))
    opts['rhols'] = float(opts0.get('rhols', 1e-6))
    opts['eta'] = float(opts0.get('eta', 0.2))
    opts['Q'] = float(opts0.get('Q', 1))

    return opts

# Default parameters for Augmented Lagrangian Method
def ALM_dual_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10))
    opts['maxit'] = int(opts0.get('maxit', 100))
    opts['maxit_inn'] = int(opts0.get('maxit_inn', 300))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    opts['thre_inn'] = float(opts0.get('thre_inn', 1e-3))
    return opts

# Default parameters for ADMM dual method
def ADMM_dual_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10))
    opts['maxit'] = int(opts0.get('maxit', 1000))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    return opts

# Default parameters for ADMM primal method
def ADMM_primal_optsInit(opts0: dict = {}):
    opts = {}
    opts['sigma'] = int(opts0.get('sigma', 10))
    opts['maxit'] = int(opts0.get('maxit', 3000))
    opts['thre'] = float(opts0.get('thre', 1e-6))
    return opts

# Initialize parameters for inner loop
def optsInnerInit(opts: dict = {}):
    optsInner = {}
    optsInner['mu0'] = float(opts.get('mu0', 1e-2))  # Target minimum mu0
    optsInner['maxit_inn'] = int(opts.get('maxit_inn', 200))
    optsInner['ftol'] = float(opts.get('ftol', 1e-8))
    optsInner['gtol'] = float(opts.get('gtol', 1e-6))
    optsInner['alpha0'] = float(opts.get('alpha0', 1))
    optsInner['gamma'] = float(opts.get('gamma', 0.9))
    optsInner['rhols'] = float(opts.get('rhols', 1e-6))
    optsInner['eta'] = float(opts.get('eta', 0.2))
    optsInner['Q'] = float(opts.get('Q', 1))
    return optsInner

def printAllDefaultOpts():
    print(f"testData: {testDataParams()}")
    print(f"gl_SGD_primal: {SGD_primal_optsInit()}")  # Default parameters for SGD primal method
    print(f"gl_ProxGD_primal: {ProxGD_primal_optsInit()}")  # Default parameters for Proximal GD method
    print(f"gl_FProxGD_primal: {FProxGD_primal_optsInit()}")  # Default parameters for Fast Proximal GD method
    print(f"gl_ALM_dual: {ALM_dual_optsInit()}")  # Default parameters for Augmented Lagrangian Method
    print(f"gl_ADMM_dual: {ADMM_dual_optsInit()}")  # Default parameters for ADMM dual method
    print(f"gl_ADMM_primal: {ADMM_primal_optsInit()}")  # Default parameters for ADMM primal method

# Initialize result output
def outInit():
    out = {}
    out['f_hist_outer'] = []  # Objective function values in each outer iteration
    out['f_hist_inner'] = []  # Objective function values in each inner iteration
    out['f_hist_best'] = []   # Best objective function values in each inner iteration
    out['g_hist'] = []        # History of gradient norms
    out['itr'] = 0            # Outer loop iteration count
    out['itr_inn'] = 0        # Total inner loop iteration count
    out['iters'] = None       # Iterator for function values
    out['fval'] = 0           # Final objective function value
    out['flag'] = False       # Convergence flag
    return out

# Continuation strategy for LASSO group problem
def LASSO_group_con(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu0: float, opts: dict = {}):
    eigs = np.linalg.eig(np.matmul(A.T, A))[0]
    eigs = np.real(eigs[np.isreal(eigs)])
    # Initialize parameters for inner loop
    optsInner = optsInnerInit(opts)
    optsInner['alpha0'] = 1 / np.max(eigs)
    optsInner['mu0'] = mu0
    optsInner['ftol'] = opts['ftol'] * opts['ftol_init_ratio']
    optsInner['gtol'] = opts['gtol'] * opts['gtol_init_ratio']
    # Initialize result output
    outResult = outInit()

    x = x0
    mu_t = opts['mu1']
    f = objFun(x, A, b, mu_t)
    solver = opts['method']
    logger.debug(f"solver: {solver} solver_name: {solver.__name__}")
    logger.info(f"optsOuter: \n{opts}")
    logger.info(f"optsInner: \n{optsInner}")

    for k in range(opts['maxit']):
        logger.debug(f"--->iter {k} : current mu_t: {mu_t}<---")
        logger.debug(f"current fval: {f}")
        logger.debug(f"current alpha0: {optsInner['alpha0']}")
        # Tighten stopping criteria
        optsInner['gtol'] = max(optsInner['gtol'] * opts['etag'], opts['gtol'])
        optsInner['ftol'] = max(optsInner['ftol'] * opts['etaf'], opts['ftol'])
        logger.debug(f"optsInner['ftol']: {optsInner['ftol']}")
        logger.debug(f"optsInner['gtol']: {optsInner['gtol']}")

        # Start inner solver
        if not callable(solver):
            logger.error(f"optsOuter['method'] is not callable")
            raise ValueError(f"optsOuter['method'] is not callable")
        fp = f
        x, itr_inn, outInner = solver(x, A, b, mu_t, optsInner)
        f = outInner['f_hist_inner'][-1]
        outResult['f_hist_inner'].extend(outInner['f_hist_inner'])
        outResult['f_hist_outer'].append(f)

        r = np.matmul(A, x) - b
        # Since L1-norm is non-differentiable, nrmG represents the violation of the optimality condition of the LASSO problem
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, r), mu0), ord="fro")
        logger.debug(f"current nrmG: {nrmG}")
        logger.debug(f"current abs(f-fp): {abs(f - fp)}")
        logger.debug(f"current itr_inn: {itr_inn}")
        logger.debug(f"is_inner_converged: {outInner['flag']}")

        # By default, mu_t is reduced in each outer iteration
        # If the inner loop converges within the specified number of iterations, do not reduce mu_t
        if outInner['flag']:
            mu_t = max(mu_t * opts['factor'], mu0)

        outResult['itr_inn'] += itr_inn  # Accumulate total inner iterations

        if mu_t == mu0 and (nrmG < opts['gtol'] or abs(f - fp) < opts['ftol']):
            logger.debug(f"--->fval has converged to {f}")
            logger.debug(f"--->nrmG has converged to {nrmG}")
            logger.debug(f"--->abs(f - fp) has converged to {abs(f - fp)}")
            break

    outResult['fval'] = f  # Final objective function value
    outResult['itr'] = k + 1  # Outer loop iteration count
    logger.debug(f"len(outResult['f_hist_inner']): {len(outResult['f_hist_inner'])}")
    logger.debug(f"outResult['itr_inn']: {outResult['itr_inn']}")
    logger.debug(f"--->end of LASSO_group_con<---")

    # Determine whether to use only outer loop iteration info
    if opts['is_only_print_outer']:
        outResult['iters'] = zip(range(outResult['itr']), outResult['f_hist_outer'])
        return x, outResult['itr'], outResult
    else:
        outResult['iters'] = zip(range(outResult['itr_inn']), outResult['f_hist_inner'])
        return x, outResult['itr_inn'], outResult