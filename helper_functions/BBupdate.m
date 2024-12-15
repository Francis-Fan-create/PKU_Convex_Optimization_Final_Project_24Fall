function alpha = BBupdate(x, xp, g, gp, k, alpha)
    % BB step size update
    dx = x - xp;
    dg = g - gp;
    dxg = abs(sum(dx(:) .* dg(:)));
    if dxg > 1e-12
        if mod(k, 2) == 1
            alpha = sum(dx(:) .* dx(:)) / dxg;
        else
            alpha = dxg / sum(dg(:) .* dg(:));
        end
    end
    alpha = max(min(alpha, 1e12), 1e-12);
end
