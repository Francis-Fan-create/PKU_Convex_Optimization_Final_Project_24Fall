import utilities.utils as utils
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set command line arguments
    parser = argparse.ArgumentParser(prog='Test_group_lasso', description='Test different solvers for group-lasso problem')
    parser.add_argument('--solvers', '-S', nargs='+', default=['gl_cvx_gurobi', 'gl_cvx_mosek'], help="Specify solver names. Input 'all' to test all solver functions in this project. Default is ['gl_cvx_gurobi', 'gl_cvx_mosek'].")
    parser.add_argument('--seed', '-RS', default=97006855, type=int, help='Specify random seed for test data. Default is 97006855.')
    parser.add_argument('--plot', '-P', action='store_true', help='Indicates whether to plot iteration curves. If this parameter is added, plotting will occur.')
    parser.add_argument('--log', '-L', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Specify log level. Default is INFO.')
    parser.add_argument('--opts', '-O', nargs='+', default={}, type=lambda kv: kv.split("="), help="Specify parameters for test data in the format 'key=value', can be multiple. For example, '-O gl_ALM_dual={'maxit':60, 'maxit_inn':100} testData={'m'=256, 'n':512}'. Parameters not specified will use default values.")
    parser.add_argument('--compare', '-C', action='store_true', help='Indicates whether to compare the computed optimal solution with the results from Mosek and Gurobi. If this parameter is added, comparison will be performed.')
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 1.0 2024-12-10')
    parser.add_argument('--printDefaultOpts', '-PDO', action='store_true', help='Show all default opts parameters.')
    args = parser.parse_args()
    utils.logger.setLevel(args.log)
    if args.printDefaultOpts:
        utils.printAllDefaultOpts()
        exit(0)
    # Process 'opts' parameter
    utils.logger.debug(f"raw opts: {args.opts}")
    if any(len(kv) < 2 for kv in args.opts):
        raise ValueError('opts parameter dictionary must be given in the form "KEY=VALUE"')
    opts = dict(args.opts)
    # Process 'solvers' parameter. If it's 'all', replace with all solvers
    if len(args.solvers) == 1 and str.lower(args.solvers[0]) == 'all':
        args.solvers = utils.solversCollection

    # Print input parameters
    utils.logger.info(f"Test solvers: {args.solvers}")
    utils.logger.info(f"opts: {opts}")
    utils.logger.info(f"random seed: {args.seed}")
    utils.logger.info(f"Is plotting?: {args.plot}")
    utils.logger.info(f"Is comparing?: {args.compare}")
    utils.logger.info(f"log level: {args.log}")

    # Initialize test data
    data_options = dict(opts.get('testData', {}))
    data_options['seed'] = args.seed
    x0, A, b, mu, u, f_u = utils.testData(data_options)
    sparsity_u = utils.sparsity(u)
    utils.logger.debug(f"Objective function value of the exact solution f_u: {f_u}")
    utils.logger.debug(f"Sparsity of the exact solution sparsity_u: {sparsity_u}")
    utils.logger.info(f"\n#######==Start all solvers TEST==#######")

    # Test results table
    table = []

    # Redirect output streams
    with open(utils.cvxLogsName, "w", encoding='utf-8') as devlog, utils.RedirectStdStreams(stdout=devlog, stderr=devlog):
        # Process 'compare' parameter
        if args.compare:
            utils.logger.info(f"\n--->Compare Solver: cvx_mosek and cvx_gurobi<---")
            x_mosek, iters_N_mosek, out_mosek = utils.testSolver(x0, A, b, mu, {'solver_name': 'gl_cvx_mosek'})
            x_gurobi, iters_N_gurobi, out_gurobi = utils.testSolver(x0, A, b, mu, {'solver_name': 'gl_cvx_gurobi'})
            table.append([
                'cvx_mosek', out_mosek['fval'], utils.errObj(out_mosek['fval'], f_u),
                utils.errX(x_mosek, u), utils.errX(x_mosek, x_mosek), utils.errX(x_mosek, x_gurobi),
                out_mosek['time_cpu'], iters_N_mosek, out_mosek['sparsity_x']])
            table.append([
                'cvx_gurobi', out_gurobi['fval'], utils.errObj(out_gurobi['fval'], f_u),
                utils.errX(x_gurobi, u), utils.errX(x_gurobi, x_mosek), utils.errX(x_gurobi, x_gurobi),
                out_gurobi['time_cpu'], iters_N_gurobi, out_gurobi['sparsity_x']])
            if args.plot:
                plot_x_mosek, plot_y_mosek = zip(*out_mosek['iters'])
                plot_x_gurobi, plot_y_gurobi = zip(*out_gurobi['iters'])
                plt.plot(plot_x_mosek, plot_y_mosek, '.-', label=('cvx_mosek in ' + str(iters_N_mosek) + ' iters'))
                plt.plot(plot_x_gurobi, plot_y_gurobi, '.-', label=('cvx_gurobi in ' + str(iters_N_gurobi) + ' iters'))
                utils.logger.info(f"Plot curve for cvx_mosek and cvx_gurobi")
        # Iterate over each solver to test
        for solver_name in args.solvers:
            # Check if solver exists
            if solver_name not in utils.solversCollection:
                utils.logger.error(f"Solver {solver_name} does not exist, skipping this solver.")
                continue
            # Check if solver has already been tested in compare
            if args.compare and solver_name in ['gl_cvx_gurobi', 'gl_cvx_mosek']:
                utils.logger.info(f"Solver {solver_name} has already been tested in compare, skipping this solver.")
                continue
            # Test solver
            solver_opts = dict(opts.get(solver_name, {}))
            solver_opts['solver_name'] = solver_name
            x, iters_N, out = utils.testSolver(x0, A, b, mu, solver_opts)
            # Process the output of the solver
            err_x_u = utils.errX(x, u)  # Compute error between x and u
            sparsity_x = utils.sparsity(x)  # Compute sparsity of x
            if iters_N == 0 or iters_N == -1:
                utils.logger.error(f"{solver_name}'s iters_N = {iters_N}, skipping this solver! Also need to check stdout redirection and log file {utils.cvxLogsName}")
                continue
            # Plot iteration curves
            plot_x, plot_y = zip(*out['iters'])
            utils.logger.debug(f"len(x)={len(plot_x)}")
            utils.logger.debug(f"len(y)={len(plot_y)}")
            if plot_y[0] < 0:  # When using Mosek to solve, y[0] may be negative. Remove it here
                plot_x = plot_x[1:]
                plot_y = plot_y[1:]
            if args.plot:
                plt.plot(plot_x, plot_y, '.-', label=(solver_name[3:] + " in " + str(iters_N) + " iters"))
                utils.logger.info(f"Plot curve for {solver_name}")
            # Create result comparison table
            if args.compare:
                table.append([solver_name[3:], out['fval'], utils.errObj(out['fval'], f_u),
                              err_x_u, utils.errX(x, x_mosek), utils.errX(x, x_gurobi),
                              out['time_cpu'], iters_N, sparsity_x])
            else:
                table.append([solver_name[3:], out['fval'], utils.errObj(out['fval'], f_u), err_x_u, out['time_cpu'], iters_N, sparsity_x])

    utils.logger.info(f"\n#######==ALL solvers have finished==#######")
    utils.logger.info(f"Objective function value of the exact solution f_u: {f_u}")
    utils.logger.info(f"Sparsity of the exact solution sparsity_u: {sparsity_u}")
    if args.compare:
        tabulate_headers = ['Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'x_CVXmosek_Error', 'x_CVXgurobi_Error', 'Time(s)', 'Iter', 'Sparsity']
    else:
        tabulate_headers = ['Solver', 'Objective', 'Obj_ABS_Error', 'x_u_Error', 'Time(s)', 'Iter', 'Sparsity']
    utils.logger.info("\n" + tabulate(table, headers=tabulate_headers))
    print("\n")
    print(tabulate(table, headers=tabulate_headers))
    if args.plot:
        plt.yscale('log')
        plt.legend()
        plt.show()


        