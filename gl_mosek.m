function [x, iter, out] = gl_mosek(x0, A, b, mu, opts)
    % Solve the group LASSO problem using MOSEK through its MATLAB interface.
    %
    % Args:
    %   x0 (matrix): Initial guess for the variable X
    %   A (matrix): Constraint matrix
    %   b (matrix): Observation vector
    %   mu (double): Regularization parameter
    %   opts (struct): Additional algorithm options (optional)
    
    % Note: for MOSEK, x0 is ignored
    [m, n] = size(A);
    l = size(b, 2);
    addpath('C:\Program Files\Mosek\10.2\toolbox\r2017a');
    % Prepare the problem structure for MOSEK
    clear prob;

    [~, res] = mosekopt('symbcon');
    % Define the linear part of the objective function (c vector)
    prob.c = zeros(n * l  + m*l + n + 1, 1); % n*l for vec(x), m*l for vec(y), 1 for t, n for z, all in column vector
    prob.c(end-n) = 0.5; % Coefficient for t
    prob.c(end-n+1:end) = mu; % Coefficients for z

    % Add linear constraint : (I_l⊗A)*vec(x)−vec(y)=vec(b)
    prob.a = sparse([kron(eye(l), A), -eye(m * l), zeros(m * l, 1), zeros(m * l, n)]);
    prob.blc = b(:);
    prob.buc = b(:);
    prob.bux = inf * ones(n * l  + m*l + n + 1, 1);
    prob.blx = -inf * ones(n * l  + m*l + n + 1, 1);
    prob.blx(end-n:end) = 0; % Bounds on z

    % Add quadratic cone constraint 1: [1+t;2*vec(y);1-t] in Q
    % FQ1 * [vec(x);vec(y);t;z] + gQ1 = [1+t;2*vec(y);1-t] in Q
    gQ1 = [1; zeros(m*l, 1); 1];
    FQ1 = sparse([zeros(1, n*l), zeros(1,m*l), 1, zeros(1, n); ...
                 zeros(m*l, n*l), 2*eye(m*l), zeros(m*l,1), zeros(m*l, n); ...
                 zeros(1, n*l), zeros(1, m*l), -1, zeros(1, n)]);
    cQ1 = [res.symbcon.MSK_DOMAIN_QUADRATIC_CONE m*l+2];

    prob.f =[FQ1];
    prob.g = [gQ1];
    prob.accs = [cQ1];

    % Add quadratic cone constraint 2: [z;vec(x(i,:))] in Q for i=1,...,n
    % Let x_i has x(i,:) as its i-th row and 0 elsewhere
    % FQ2_i * [vec(x);vec(y);t;z] + gQ2_i = [z;vec(x_i)] in Q
    for i=1:n
        elementary_i = zeros(n,n);
        elementary_i(i,i) = 1;
        elementary_i_diag_cat = kron(eye(l), elementary_i);
        elementary_i_upper = zeros(n,n);
        elementary_i_upper(1,i) = 1;
        gQ2_i = zeros(n+n*l, 1);
        FQ2_i = sparse([zeros(n,n*l+m*l+1) elementary_i_upper; elementary_i_diag_cat zeros(n*l,m*l+1+n)]);
        cQ2_i = [res.symbcon.MSK_DOMAIN_QUADRATIC_CONE n+n*l];
        prob.f = [prob.f; FQ2_i];
        prob.g = [prob.g; gQ2_i];
        prob.accs = [prob.accs cQ2_i];
    end




    % Solve the problem using MOSEK
    [~, res] = mosekopt('minimize', prob);

    % Prepare output structure
    out = struct();
    out.fval = res.sol.itr.pobjval;
    out.iter = -1;
    iter = out.iter;

    % Extract solution
    x = reshape(res.sol.itr.xx(1:n*l), n, l);


end