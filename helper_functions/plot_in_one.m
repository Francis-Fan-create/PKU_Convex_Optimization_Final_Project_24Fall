function plot_in_one(title, filename, varargin)
    % Check if out.iter==-1
    figure;
    for i = 1:length(varargin)
        out = varargin{i};
        if out.iter == -1
            fprintf('The solver did provide error history.\n');
        else
            plot_x = out.iters(:,1);
            plot_y = out.iters(:,2);
            plot(plot_x, plot_y, "LineWidth", 1.5);
            hold on;
        end
    end
    hold off;
    xlabel('Iteration');
    ylabel('Objective Value');
    set(gca, 'YScale', 'log');
    legend('CVX-Mosek', 'CVX-Gurobi', 'Mosek', 'Gurobi', 'SGD Primal', 'ProxGD Primal', 'FProxGD Primal', 'ALM Dual', 'ADMM Dual', 'ADMM Primal');
    saveas(gcf, filename);
end