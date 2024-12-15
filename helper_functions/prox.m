function prox_x = prox(x, mu)
    % Proximal operator
    norm_x = vecnorm(x, 2, 2);
    flag = norm_x > mu;
    prox_x = x- mu * x ./ (norm_x + 1e-10);
    prox_x = prox_x .* flag;
end