function e = errX(x, x0)
    % Calculate the difference between x and x0
    e = norm(x - x0, 'fro') / (1 + norm(x0, 'fro'));
end