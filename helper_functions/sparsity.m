function s = sparsity(x)
    % Calculate sparsity of x
    s = sum(abs(x(:)) > 1e-5) / numel(x);
end










    

















