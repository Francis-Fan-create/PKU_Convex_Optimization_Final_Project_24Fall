function [x, iter, output] = gl_ALM_dual(x0, A, b, mu, opts)
    % Solve the dual problem of the group LASSO using the Augmented Lagrangian Method (ALM).
    %
    % Args:
    %     x0 (matrix): Initial guess for the variable x.
    %     A (matrix): Constraint matrix.
    %     b (matrix): Observation vector.
    %     mu (float): Regularization parameter.
    %     opts (struct, optional): Algorithm options.
    %
    % Returns:
    %     x (matrix): Updated x.
    %     iter (int): Number of ks.
    %     output (struct): Output information.

    % Initialize options with default ALM dual parameters
    opts = ALM_dual_optsInit(opts);

    [m, n] = size(A);
    [~, l] = size(b);
    output = outInit();
    output.prim_hist = [];
    output.dual_hist = [];
    output.itr_inn = 0;

    x = x0;
    sigma = opts.sigma;   % Quadratic penalty coefficient
    z = zeros(n, l);

    for outer_iter = 1:opts.maxit
        for inner_iter = 1:opts.maxit_inn
            % Update dual variable y
            y = (eye(m) + sigma * (A * A'))\(A * x - sigma * A * z - b);

            % Store previous z for convergence check
            previous_z = z;

            % Update primal variable z using the proximal operator
            z = updateZ(x ./ sigma - A' * y, mu);

            % Calculate the change in z for convergence
            z_change = norm(z - previous_z, 'fro');

            % Increment total inner ks
            output.itr_inn = output.itr_inn + 1;

            % Check convergence of the inner loop
            if z_change < opts.thre_inn
                break;
            end
        end

        % Update primal variable x
        x = x - sigma * (A' * y + z);

        % Compute the primal objective function value
        primal_obj = objFun(x, A, b, mu);

        % Compute the dual objective function value
        dual_obj = 0.5 * norm(y, 'fro')^2 - sum(y .* b);

        % Record history of objective values
        output.prim_hist = [output.prim_hist; primal_obj];
        output.dual_hist = [output.dual_hist; dual_obj];

        % Check convergence based on the norm of the r
        if norm(A' * y + z) < opts.thre
            break;
        end
    end

    % Store the number of ks and final objective value
    output.iter = outer_iter;
    output.fval = primal_obj;

    % Prepare k history for output
    output.iters = [(1:output.iter)', output.prim_hist];

    iter = output.iter;
end

function z_updated = updateZ(reference, mu)
    % Update the variable Z using the proximal operator.

    % Compute the norms of each row
    norm_ref = sqrt(sum(reference .^ 2, 2));

    % Adjust norms less than mu to mu
    norm_ref_adj = norm_ref;
    norm_ref_adj(norm_ref_adj < mu) = mu;

    % Compute updated z
    z_updated = reference .* (mu ./ norm_ref_adj);
end


