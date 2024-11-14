import utilities as util
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

def setup_arguments():
    parser = argparse.ArgumentParser(
        program='GroupLassoTester',
        description='Evaluate various solvers for the group LASSO optimization task'
    )
    parser.add_argument(
        '--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'],
        help='List of solvers to evaluate. Use `all` to run every solver available. Defaults to [\'gl_cvx_gurobi\', \'gl_cvx_mosek\'].'
    )
    parser.add_argument(
        '--seed', '-RS', default=1, type=int,
        help='Random seed for generating test data. Default value is 1.'
    )
    parser.add_argument(
        '--plot', '-P', action='store_true',
        help='Enable plotting of iteration progress. Plots will be generated if this flag is set.'
    )
    parser.add_argument(
        '--log', '-L', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level. Default is INFO.'
    )
    parser.add_argument(
        '--options', '-O', nargs='+', default={}, type=lambda kv: kv.split("="),
        help='Define test parameters in the format `key=value`. Multiple entries allowed, e.g., `-O solver_opts={\'max_iters\':60} data_params={\'dim_m\':256, \'dim_n\':512}`. Unspecified parameters retain their default settings.'
    )
    parser.add_argument(
        '--compare', '-C', action='store_true',
        help='Compare optimal solutions between Mosek and Gurobi solvers when this flag is enabled.'
    )
    parser.add_argument(
        '--version', '-V', action='version', version='GroupLassoTester 1.0 2024-11-15'
    )
    parser.add_argument(
        '--showDefaults', '-PDO', action='store_true',
        help='Print all default option settings.'
    )
    return parser.parse_args()

def main():
    args = setup_arguments()
    util.logger.setLevel(args.log)
    
    if args.showDefaults:
        util.displayDefaultOptions()
        exit(0)
    
    util.logger.debug(f"Received options: {args.options}")
    if any(len(pair) < 2 for pair in args.options):
        raise ValueError('Options must follow the "KEY=VALUE" format.')
    
    configurations = dict(args.options)
    
    if len(args.solvers) == 1 and args.solvers[0].lower() == 'all':
        args.solvers = util.availableSolvers
    
    util.logger.info(f"Solvers under test: {args.solvers}")
    util.logger.info(f"Configurations: {configurations}")
    util.logger.info(f"Random seed set to: {args.seed}")
    util.logger.info(f"Plotting enabled: {args.plot}")
    util.logger.info(f"Comparison enabled: {args.compare}")
    util.logger.info(f"Logging level set to: {args.log}")
    
    test_params = dict(configurations.get('data_params', {}))
    test_params['seed'] = args.seed
    initial_vals, matrix_A, vector_b, regularization_mu, solution_u, exact_f = util.generateTestData(test_params)
    exact_sparsity = util.calculateSparsity(solution_u)
    
    util.logger.debug(f"Exact objective value: {exact_f}")
    util.logger.debug(f"Exact solution sparsity: {exact_sparsity}")
    util.logger.info("\n#######==Initiating Solver Tests==#######")
    
    results_table = []
    
    with open(util.logFileName, "w", encoding='utf-8') as log_file, util.RedirectOutput(stdout=log_file, stderr=log_file):
        if args.compare:
            util.logger.info("\n--->Comparing cvx_mosek and cvx_gurobi Solvers<---")
            mosek_result = util.runSolver(initial_vals, matrix_A, vector_b, regularization_mu, {'solver': 'gl_cvx_mosek'})
            gurobi_result = util.runSolver(initial_vals, matrix_A, vector_b, regularization_mu, {'solver': 'gl_cvx_gurobi'})
            
            results_table.extend([
                [
                    'cvx_mosek', mosek_result['objective'], util.objectiveError(mosek_result['objective'], exact_f),
                    util.calculateError(mosek_result['solution'], solution_u), 
                    util.calculateError(mosek_result['solution'], mosek_result['solution']), 
                    util.calculateError(mosek_result['solution'], gurobi_result['solution']), 
                    mosek_result['cpu_time'], mosek_result['iterations'], mosek_result['sparsity']
                ],
                [
                    'cvx_gurobi', gurobi_result['objective'], util.objectiveError(gurobi_result['objective'], exact_f),
                    util.calculateError(gurobi_result['solution'], solution_u), 
                    util.calculateError(gurobi_result['solution'], mosek_result['solution']), 
                    util.calculateError(gurobi_result['solution'], gurobi_result['solution']),
                    gurobi_result['cpu_time'], gurobi_result['iterations'], gurobi_result['sparsity']
                ]
            ])
            
            if args.plot:
                mosek_iters_x, mosek_iters_y = zip(*mosek_result['iterations_data'])
                gurobi_iters_x, gurobi_iters_y = zip(*gurobi_result['iterations_data'])
                plt.plot(mosek_iters_x, mosek_iters_y, '.-', label=f'cvx_mosek ({mosek_result["iterations"]} iterations)')
                plt.plot(gurobi_iters_x, gurobi_iters_y, '.-', label=f'cvx_gurobi ({gurobi_result["iterations"]} iterations)')
                util.logger.info("Generated iteration plots for cvx_mosek and cvx_gurobi")
        
        for solver in args.solvers:
            if solver not in util.availableSolvers:
                util.logger.error(f"Solver {solver} is unavailable. Skipping.")
                continue
            if args.compare and solver in ['gl_cvx_gurobi', 'gl_cvx_mosek']:
                util.logger.info(f"Solver {solver} already evaluated during comparison. Skipping.")
                continue
            
            solver_options = dict(configurations.get(solver, {}))
            solver_options['solver'] = solver
            result = util.runSolver(initial_vals, matrix_A, vector_b, regularization_mu, solver_options)
            
            solution_error = util.calculateError(result['solution'], solution_u)
            solution_sparsity = util.calculateSparsity(result['solution'])
            
            if result['iterations'] in [0, -1]:
                util.logger.error(
                    f"{solver} returned {result['iterations']} iterations. Skipping this solver! Check log file {util.logFileName}"
                )
                continue
            
            iter_x, iter_y = zip(*result['iterations_data'])
            util.logger.debug(f"Iteration steps count: {len(iter_x)}, Iteration values count: {len(iter_y)}")
            if iter_y[0] < 0:
                iter_x = iter_x[1:]
                iter_y = iter_y[1:]
            
            if args.plot:
                plt.plot(iter_x, iter_y, '.-', label=f"{solver[3:]} ({result['iterations']} iterations)")
                util.logger.info(f"Generated iteration plot for {solver}")
            
            if args.compare:
                results_table.append([
                    solver[3:], result['objective'], util.objectiveError(result['objective'], exact_f),
                    solution_error, util.calculateError(result['solution'], mosek_result['solution']),
                    util.calculateError(result['solution'], gurobi_result['solution']),
                    result['cpu_time'], result['iterations'], solution_sparsity
                ])
            else:
                results_table.append([
                    solver[3:], result['objective'], util.objectiveError(result['objective'], exact_f),
                    solution_error, result['cpu_time'], result['iterations'], solution_sparsity
                ])
    
    util.logger.info("\n#######==All Solver Evaluations Completed==#######")
    util.logger.info(f"Exact Objective Value: {exact_f}")
    util.logger.info(f"Exact Solution Sparsity: {exact_sparsity}")
    
    headers = [
        'Solver', 'Objective', 'Objective Error', 'Solution Error', 
        'Time (s)', 'Iterations', 'Sparsity'
    ]
    if args.compare:
        headers = [
            'Solver', 'Objective', 'Objective Error', 'Solution Error', 
            'Error vs Mosek', 'Error vs Gurobi', 'Time (s)', 'Iterations', 'Sparsity'
        ]
    
    table_output = tabulate(results_table, headers=headers)
    util.logger.info("\n" + table_output)
    print("\n" + table_output)
    
    if args.plot:
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()