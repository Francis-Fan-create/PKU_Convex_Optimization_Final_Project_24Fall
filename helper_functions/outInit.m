function out = outInit()
    % Initialize result output
    out.f_hist_outer = []; % Objective function values in each outer k
    out.f_hist_inner = []; % Objective function values in each inner k
    out.f_hist_best = [];  % Best objective function values in each inner k
    out.g_hist = [];       % History of g norms
    out.iter = 0;           % Outer loop k count
    out.itr_inn = 0;       % Total inner loop k count
    out.iters = [];        % Iterator for function values
    out.fval = 0;          % Final objective function value
    out.flag = false;      % Convergence flag
end