function [Rxxs, Eun, w, bun] = maxRESMACMIMO(H, Lxu, Etotal, theta, cb, use_mex)
    %MAXRESMACMIMO Maximize weighted rate sum subject to total energy constraint
    %   [Rxxs, Eun, w, bun] = MAXRESMACMIMO(H, Lxu, Etotal, theta, cb) computes
    %   the optimal covariance matrices that maximize the weighted rate sum
    %   subject to a total sum-energy constraint for MIMO MAC.
    %
    %   [Rxxs, Eun, w, bun] = MAXRESMACMIMO(..., use_mex) uses the C++ MEX
    %   implementation when use_mex is true. Default is true.
    %
    %   Inputs:
    %       H       channel matrix [Ly, sum(Lxu), N] concatenated across users.
    %       Lxu     antennas per user [1, U] where sum(Lxu) equals total TX antennas.
    %               If scalar, all users have the same number of antennas.
    %       Etotal  total sum-energy constraint (scalar).
    %       theta   rate weights per user [U, 1] or [1, U].
    %       cb      baseband type: 1 for complex, 2 for real (affects rate scaling).
    %       use_mex (optional) use C++ MEX implementation (default: true).
    %
    %   Outputs:
    %       Rxxs    U-by-N cell array of Rxx(u,n) matrices, or Lxu_max-by-Lxu_max-by-U-by-N
    %               tensor if Lxu is scalar (uniform antennas).
    %       Eun     U x N energy distribution matrix.
    %       w       U x 1 Lagrangian multiplier vector w.r.t. energy constraints.
    %       bun     U x N bit distribution matrix for all users.
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    if nargin < 6
        use_mex = true;
    end

    if use_mex && exist('maxresmac_mex', 'file') == 3
        [~, Ltot, ~] = size(H);
        theta = reshape(theta, [], 1);
        U = length(theta);

        if isscalar(Lxu)
            Lxu_vec = ones(1, U) * Lxu;
            uniform_flag = true;
        else
            Lxu_vec = reshape(Lxu, 1, []);
            uniform_flag = false;
        end

        if isreal(H)
            H = complex(H);
        end

        [Rxx_cell, Eun, w, bun] = maxresmac_mex(H, Lxu_vec, sum(Etotal(:)), theta, cb);

        N = size(bun, 2);
        Lxu_max = max(Lxu_vec);

        if uniform_flag
            Rxxs = zeros(Lxu_max, Lxu_max, U, N);

            for u = 1:U

                for n = 1:N
                    Rxxs(1:Lxu_vec(u), 1:Lxu_vec(u), u, n) = Rxx_cell{u, n};
                end

            end

        else
            Rxxs = Rxx_cell;
        end

        if uniform_flag
            Rxxs = real(Rxxs);
        else
            Rxxs = cellfun(@real, Rxxs, 'UniformOutput', false);
        end

        return;
    end

    this_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(this_dir, 'utils'));

    [Ly, Ltot, N] = size(H);

    % handle real baseband symmetry
    if N > 1 && cb == 2
        H = H(:, :, 1:N / 2);
        N = N / 2;
    end

    if N == 1
        H = reshape(H, Ly, [], 1);
    end

    Etotal = sum(Etotal(:)); % ensure scalar
    theta = reshape(theta, [], 1);
    U = length(theta);

    UNIFORM_FLAG = false;

    if isscalar(Lxu)
        Lxu = ones(1, U) * Lxu;
        UNIFORM_FLAG = true;
    else
        Lxu = reshape(Lxu, 1, []);
    end

    if sum(Lxu) ~= Ltot
        error('sum(Lxu) = %d must equal total TX antennas = %d', sum(Lxu), Ltot);
    end

    Lxu_max = max(Lxu);

    % compute antenna index mapping
    idx_end = cumsum(Lxu);
    idx_start = [1, idx_end(1:end - 1) + 1];

    % determine decoding order (descending theta)
    [stheta, order] = sort(theta, 'descend');
    delta = stheta - [stheta(2:end); 0];

    % map parameters to sorted order
    Lxu_sorted = Lxu(order);
    idx_start_sorted = idx_start(order);
    idx_end_sorted = idx_end(order);

    % bisection on lambda to find the value that uses exactly Etotal
    lambda_min = 1e-14;
    lambda_max = 1e14;
    tol = 1e-8;
    max_iter = 120;

    best_Rxx = [];
    best_E = [];
    best_lambda = 1.0;

    for iter = 1:max_iter
        lambda = sqrt(lambda_min * lambda_max); % geometric mean

        [Rxx_sorted, E_sorted] = eval_lagfunc(H, Lxu_sorted, idx_start_sorted, idx_end_sorted, delta, lambda, U, N, Ly);

        E_current = sum(E_sorted, 'all');

        if abs(E_current - Etotal) / max(Etotal, 1e-12) < tol
            best_Rxx = Rxx_sorted;
            best_E = E_sorted;
            best_lambda = lambda;
            break;
        end

        if E_current > Etotal
            lambda_min = lambda;
        else
            lambda_max = lambda;
        end

        best_Rxx = Rxx_sorted;
        best_E = E_sorted;
        best_lambda = lambda;
    end

    Rxx_sorted = best_Rxx;
    E_sorted = best_E;

    % compute rates from the obtained covariance matrices
    bun_sorted = compute_rates(H, Rxx_sorted, Lxu_sorted, idx_start_sorted, idx_end_sorted, cb, U, N, Ly);

    % map results back to original user ordering
    inv_order = zeros(U, 1);

    for u = 1:U
        inv_order(order(u)) = u;
    end

    Eun = zeros(U, N);
    bun = zeros(U, N);

    for u = 1:U
        Eun(u, :) = E_sorted(inv_order(u), :);
        bun(u, :) = bun_sorted(inv_order(u), :);
    end

    % output lagrange multiplier
    w = best_lambda * ones(U, 1);

    % assemble output covariance matrices
    if UNIFORM_FLAG
        Rxxs = zeros(Lxu_max, Lxu_max, U, N);

        for u = 1:U
            sorted_u = inv_order(u);

            for n = 1:N
                Lu = Lxu_sorted(sorted_u);
                Rxxs(1:Lu, 1:Lu, u, n) = Rxx_sorted{sorted_u, n};
            end

        end

    else
        Rxxs = cell(U, N);

        for u = 1:U
            sorted_u = inv_order(u);

            for n = 1:N
                Rxxs{u, n} = Rxx_sorted{sorted_u, n};
            end

        end

    end

end

function [Rxx_all, E_all] = eval_lagfunc(H, Lxu, idx_start, idx_end, delta, lambda, U, N, Ly)
    %EVAL_LAGFUNC Evaluate Lagrangian dual by solving each tone
    %   Calls solve_tone for each frequency tone and aggregates results.

    Rxx_all = cell(U, N);
    E_all = zeros(U, N);

    for n = 1:N
        [Rxx_n, E_n] = solve_tone(H(:, :, n), Lxu, idx_start, idx_end, delta, lambda, U, Ly);

        for u = 1:U
            Rxx_all{u, n} = Rxx_n{u};
            E_all(u, n) = E_n(u);
        end

    end

end

function bun = compute_rates(H, Rxx_all, Lxu, idx_start, idx_end, cb, U, N, Ly)
    %COMPUTE_RATES Compute per-user per-tone rates from covariance matrices

    bun = zeros(U, N);
    eye_Ly = eye(Ly);
    inv_log2_cb = 1 / (log(2) * cb);

    for n = 1:N
        Hn = H(:, :, n);
        S_prev = eye_Ly;

        for u = 1:U
            cols = idx_start(u):idx_end(u);
            Hu = Hn(:, cols);
            HRH = Hu * Rxx_all{u, n} * Hu';
            HRH = 0.5 * (HRH + HRH');

            S_curr = S_prev + HRH;
            S_curr = 0.5 * (S_curr + S_curr');

            logdet_prev = logdet_stable(S_prev);
            logdet_curr = logdet_stable(S_curr);

            bun(u, n) = (logdet_curr - logdet_prev) * inv_log2_cb;
            S_prev = S_curr;
        end

    end

end

function val = logdet_stable(M)
    %LOGDET_STABLE Numerically stable log-determinant computation

    M = 0.5 * (M + M');
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-12 * eye(size(M, 1));
        [L, flag] = chol(M, 'lower');

        if flag ~= 0
            eigvals = real(eig(M));
            eigvals = max(eigvals, 1e-15);
            val = sum(log(eigvals));
            return;
        end

    end

    val = 2 * sum(log(diag(L)));
end
