% Finish the plot for following commands:
% plot_results(u, 'Exact', '../figures/gl_exact.png', u, x1, x2)
% plot_results(x1, 'CVX-Mosek', '../figures/gl_cvx_mosek.png', u, x1, x2)
% plot_results(x2, 'CVX-Gurobi', '../figures/gl_cvx_gurobi.png', u, x1, x2)
% plot_results(x3, 'Mosek', '../figures/gl_mosek.png', u, x1, x2)
% plot_results(x4, 'Gurobi', '../figures/gl_gurobi.png', u, x1, x2)
% plot_results(x5, 'SGD Primal', '../figures/gl_SGD_Primal.png', u, x1, x2)
% plot_results(x8, 'ProxGD Primal', '../figures/gl_ProxGD_primal.png', u, x1, x2)
% plot_results(x9, 'FProxGD Primal', '../figures/gl_FProxGD_primal.png', u, x1, x2)
% plot_results(x10, 'ALM Dual', '../figures/gl_ALM_dual.png', u, x1, x2)
% plot_results(x11, 'ADMM Dual', '../figures/gl_ADMM_dual.png', u, x1, x2)
% plot_results(x12, 'ADMM Primal', '../figures/gl_ADMM_primal.png', u, x1, x2)
function plot_results(title, filename, out)
    % Check if out.iter==-1
    if out.iter == -1
        fprintf('Solver Name: %s\n', title);
        fprintf('The solver did provide error history.\n');
    else
        plot_x = out.iters(:,1);
        plot_y = out.iters(:,2);
        % Plot the objective function value in red
        plot(plot_x, plot_y, 'Color', 'r','LineWidth', 1.5);
        xlabel('Iteration');
        ylabel('Objective Value');
        lengend_string = sprintf('%s: %d iterations', title, out.iter);
        legend(lengend_string, 'Location', 'northeast');
        set(gca, 'YScale', 'log');
        saveas(gcf, filename);
        close;
    end
end