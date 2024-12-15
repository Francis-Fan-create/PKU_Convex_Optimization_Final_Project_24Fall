function f = objFun(x, A, b, mu)
    % Calculate the objective function value
    f = 0.5 * norm(A * x - b, 'fro') ^ 2 + mu * sum(vecnorm(x, 2, 2));
end
