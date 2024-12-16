# PKU_Convex_Optimization_Final_Project_24
A matlab implementation of the group LASSO problem at Optimization Methods taught by Zaiwen Wen at PKU, 24 Fall.

## Problem Formulation

Consider group LASSO problem

$$
\min_{x\in\mathbb{R}^{n\times l}}\quad\frac{1}{2}\|Ax-b\|_F^2+\mu\|x\|_{1,2}
$$

Here:

$$
\begin{align}
A&\in\mathbb{R}^{m\times n}\\
b&\in\mathbb{R}^{m\times l}\\
\mu&>0
\end{align}
$$

and that

$$
\|x\|_{1,2}=\sum_{i=1}^n\|x(i,1:l)\|_2.
$$

where $x(i,1:l),1\leq i\leq n$ is the $i$-th row of matrix $x$.

## Usage

1. Change the working dir to this repo
1. Change the Gurobi/Mosek dir in `gl_gurobi.m`/`gl_mosek.m` to your own correspondingly
2. Run `Test_group_lasso.m`

## Result Overview

![gl_compare](figures\gl_compare.png)

| Method         | CPU   | Iter | Optval      | Sparsity | Err-to-Exact | Err-to-CVX-Mosek | Err-to-CVX-Gurobi |
| -------------- | ----- | ---- | ----------- | -------- | ------------ | ---------------- | ----------------- |
| CVX-Mosek      | 1.45  | -1   | 5.38327E-01 | 0.115    | 4.21E-05     | 0.00E+00         | 7.49E-08          |
| CVX-Gurobi     | 0.64  | -1   | 5.38327E-01 | 0.115    | 4.21E-05     | 7.49E-08         | 0.00E+00          |
| Mosek          | 12.90 | -1   | 5.38327E-01 | 0.114    | 4.20E-05     | 9.87E-08         | 7.25E-08          |
| Gurobi         | 15.73 | -1   | 5.38327E-01 | 0.115    | 4.22E-05     | 2.17E-07         | 2.50E-07          |
| SGD Primal     | 0.39  | 1871 | 5.38331E-01 | 0.164    | 6.33E-05     | 2.59E-05         | 2.59E-05          |
| ProxGD Primal  | 0.06  | 185  | 5.38327E-01 | 0.114    | 4.20E-05     | 1.65E-07         | 1.33E-07          |
| FProxGD Primal | 0.07  | 440  | 5.38327E-01 | 0.114    | 4.20E-05     | 1.74E-07         | 1.44E-07          |
| ALM Dual       | 0.30  | 61   | 5.38345E-01 | 0.100    | 7.84E-05     | 4.41E-05         | 4.41E-05          |
| ADMM Dual      | 0.09  | 86   | 5.38341E-01 | 0.100    | 7.50E-05     | 3.84E-05         | 3.84E-05          |
| ADMM Primal    | 8.80  | 2452 | 5.38327E-01 | 0.114    | 4.20E-05     | 2.11E-07         | 1.90E-07          |

## Solver Description

Each solver is in `gl_(solver's name).m`  format, and all of them can be called separately by the following format:

```matlab
[x, iter, out] = gl_solver_name(x0, A, b, mu, opts)
```

Here:

- `x0`: initial solution
- `A ,b ,mu`: given parameters from the problem
- `opts`: a structure, containing input parameters for each solver. Check `(solver's name)_optsInit.m` for their default values
- `x`: final solution from the solver
- `iter`: iterations when `x` is outputed
- `out`: a structure, containing additional information of the solving process. Including:
  - `fval`: objective value at `x`
  - `iters`: a enumerated list of objective value at each step
  - Other informations specific for each solvers

## Reference

[repo: group-lasso-optimization](https://github.com/gzz2000/group-lasso-optimization)

[Matlab examples from class](http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/contents/contents.html)

[repo: Algorithms-group-LASSO-problem](https://github.com/AkexStar/Algorithms-group-LASSO-problem/tree/main)
