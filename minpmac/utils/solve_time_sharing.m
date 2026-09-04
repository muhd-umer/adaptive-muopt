function [weights, active_idx] = solve_time_sharing(bu_vertices, bu_min, tol)
    %SOLVE_TIME_SHARING Solve time-sharing via linear programming (no CVX)
    %   [weights, active_idx] = SOLVE_TIME_SHARING(bu_vertices, bu_min, tol)
    %   finds non-negative convex weights such that the weighted combination
    %   of rate vertices meets target rates. Uses MATLAB's linprog for LP.
    %
    %   Inputs:
    %       bu_vertices  U x K matrix of achievable rates for each decoding order.
    %       bu_min       U x 1 vector of minimum target rates.
    %       tol          tolerance for rate constraints (default 1e-3).
    %
    %   Outputs:
    %       weights      K x 1 vector of time-sharing weights (empty if infeasible).
    %       active_idx   logical K x 1 vector indicating which orders are used.
    %
    %   The LP minimizes sum of weights subject to:
    %       bu_vertices * weights >= bu_min - tol
    %       sum(weights) == 1
    %       weights >= 0
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    if nargin < 3
        tol = 1e-3;
    end

    [U, K] = size(bu_vertices);
    bu_min = bu_min(:);

    % [phase 1] find feasible weights using LP
    % minimize: 0 (feasibility problem)
    % subject to: bu_vertices * w >= bu_min - tol
    %             sum(w) == 1
    %             w >= 0
    %
    % convert to standard LP form for linprog:
    %   min f'*x  s.t. A*x <= b, Aeq*x = beq, lb <= x <= ub
    %
    % inequality: -bu_vertices * w <= -bu_min + tol
    %   A is U x K, w is K x 1, b is U x 1
    % equality: sum(w) = 1

    f = zeros(K, 1);
    A = -bu_vertices;
    b = -bu_min + tol;
    Aeq = ones(1, K);
    beq = 1;
    lb = zeros(K, 1);
    ub = ones(K, 1);

    options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');

    [w_feas, ~, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);

    if exitflag ~= 1
        options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'interior-point');
        [w_feas, ~, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);
    end

    if exitflag ~= 1
        weights = [];
        active_idx = [];
        return;
    end

    % [phase 2] sparsify the solution by minimizing L1-norm of weights
    % this tends to produce sparse solutions with fewer active orderings
    % minimize: sum(w) = 1 (already enforced), so we minimize the number
    % of non-zero entries by using L1 regularization on auxiliary slack

    thresh = 1e-6;
    w_feas(w_feas < thresh) = 0;

    if sum(w_feas) > 0
        w_feas = w_feas / sum(w_feas);
    else
        weights = [];
        active_idx = [];
        return;
    end

    active_idx = w_feas > thresh;

    weights = w_feas(active_idx);
    weights = weights / sum(weights);
end
