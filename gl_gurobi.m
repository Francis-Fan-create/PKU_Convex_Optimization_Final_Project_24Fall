function [X, iter, out] = gl_gurobi(x0, A, b, mu, opts)
    % Initialize model
    [m, n] = size(A);
    l = size(b, 2);
    addpath('C:\Users\86189\gurobi\win64\matlab');
    model = struct();
    % variables: x11, x21, ..., xn1, x12, x22, ..., xn2, ..., x1l, x2l, ..., xnl, y11, y21, ..., ym1, y12, y22, ..., ym2, ..., yml, z1, z2, ..., zn
    model.varnames = cell(1, n*l + m*l + n );
    for j=1:l
        for i=1:n
            model.varnames{(j-1)*n+i} = ['x' num2str(i) num2str(j)];
        end
    end
    for j=1:l
        for i=1:m
            model.varnames{l*n+(j-1)*m+i} = ['y' num2str(i) num2str(j)];
        end
    end
    for i=1:n
        model.varnames{n*l+m*l+i} = ['z' num2str(i)];
    end

    % Objective: 0.5 * sum(y.^2) + mu * sum(z)
    % Define the quad part
    quadrature = zeros(n*l+m*l+n, n*l+m*l+n);
    quadrature(n*l+1:n*l+m*l, n*l+1:n*l+m*l) = 0.5*eye(m*l);
    model.Q = sparse(quadrature);
    % Define the linear part
    model.obj = zeros(1, n*l+m*l+n);
    model.obj(n*l+m*l+1:end) = mu;

    % Linear constraints: (I_l⊗A)*vec(x)−vec(y)=vec(b)
    Aeq = kron(eye(l), A);
    beq = b(:);
    model.A = sparse([Aeq, -eye(m*l),zeros(m*l, n)]);
    model.rhs = beq;
    model.sense = '=';

    % Bounds on t,z, make it non-negative
    model.lb = -inf*ones(n*l+m*l+n, 1);
    model.ub = inf*ones(n*l+m*l+n, 1);
    model.lb(n*l+m*l+1:end) = 0;

    % Quadratic cone constraints: [zi;xi1;xi2;...;xil] in Q for i=1,...,n
    for i=1:n
        elementary_i = zeros(n,n);
        elementary_i(i,i) = 1;
        elementary_i_diag_cat = kron(eye(l), elementary_i);
        constraint_matrix = sparse([elementary_i_diag_cat, zeros(n*l,m*l+n); zeros(m*l,n*l+m*l+n); zeros(n,n*l+m*l), -elementary_i]);
        model.quadcon(i).Qc = constraint_matrix;
        model.quadcon(i).q = zeros(n*l+m*l+n, 1);
        model.quadcon(i).rhs = 0;
        model.quadcon(i).name = ['QC' num2str(i)];
    end

    result = gurobi(model);
    X = reshape(result.x(1:n*l),n,l);
    iter = -1;
    out = struct();
    out.fval = result.objval;
    out.iter = iter;
end