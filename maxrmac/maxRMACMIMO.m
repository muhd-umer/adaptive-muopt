function [Rxxs, Eun, w, bun] = maxRMACMIMO(H, Lxu, Eu, theta, cb, use_mex)
    %MAXRMACMIMO Maximize weighted rate sum with per-user energy constraints
    %   [Rxxs, Eun, w, bun] = MAXRMACMIMO(H, Lxu, Eu, theta, cb) maximizes the
    %   weighted sum rate for a (noise-whitened) multi-user MIMO MAC subject to
    %   per-user total energy constraints.
    %
    %   [Rxxs, Eun, w, bun] = MAXRMACMIMO(..., use_mex) uses the C++ MEX
    %   implementation when use_mex is true. Default is true.
    %
    %   This uses the ellipsoid method to optimize the Lagrange dual over energy
    %   multipliers and L-BFGS to solve the per-tone inner maximization via
    %   Cholesky-like factors.
    %
    %   Inputs:
    %       H       channel tensor [Ly, sum(Lxu), N] with user antennas stacked.
    %               For SISO, sum(Lxu)=U and H is Ly-by-U-by-N.
    %       Lxu     antennas per user (scalar or 1-by-U).
    %       Eu      per-user energy constraints (length-U vector).
    %       theta   per-user rate weights (length-U vector).
    %       cb      baseband type: 1 for complex, 2 for real.
    %       use_mex (optional) use C++ MEX implementation (default: true).
    %
    %   Outputs:
    %       Rxxs    U-by-N cell array of covariance matrices, or
    %               Lxu_max-by-Lxu_max-by-U-by-N tensor if Lxu is scalar.
    %       Eun     U-by-N energy allocation, Eun(u,n)=trace(Rxx(u,n)).
    %       w       U-by-1 dual variables for energy constraints.
    %       bun     U-by-N per-user per-tone rates (bits).
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    if nargin < 6
        use_mex = true;
    end

    if use_mex

        if exist('maxrmac_mex', 'file') ~= 3
            error('maxRMACMIMO:mexNotFound', ...
            'MEX file ''maxrmac_mex'' not found. Compile it first or set use_mex=false.');
        end

        [Ly, Ltot, ~] = size(H);
        theta = reshape(theta, [], 1);
        Eu = reshape(Eu, [], 1);
        U = length(theta);

        if length(Eu) ~= U
            error('Eu must have length U=%d', U);
        end

        if cb ~= 1 && cb ~= 2
            error('cb must be 1 (complex) or 2 (real)');
        end

        if isscalar(Lxu)
            Lxu_vec = ones(1, U) * Lxu;
            uniform_flag = true;
        else
            Lxu_vec = reshape(Lxu, 1, []);
            uniform_flag = false;
        end

        if length(Lxu_vec) ~= U
            error('Lxu must be a scalar or a length-U vector');
        end

        if sum(Lxu_vec) ~= Ltot
            error('sum(Lxu)=%d must equal size(H,2)=%d', sum(Lxu_vec), Ltot);
        end

        if isreal(H)
            H = complex(H);
        end

        [Rxx_cell, Eun, w, bun] = maxrmac_mex(H, Lxu_vec, Eu, theta, cb);

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

    [Ly, Ltot, N_in] = size(H);
    theta = reshape(theta, [], 1);
    Eu = reshape(Eu, [], 1);
    U = length(theta);

    if length(Eu) ~= U
        error('Eu must have length U=%d', U);
    end

    if cb ~= 1 && cb ~= 2
        error('cb must be 1 (complex) or 2 (real)');
    end

    if isscalar(Lxu)
        Lxu_vec = ones(1, U) * Lxu;
        uniform_flag = true;
    else
        Lxu_vec = reshape(Lxu, 1, []);
        uniform_flag = false;
    end

    if length(Lxu_vec) ~= U
        error('Lxu must be a scalar or a length-U vector');
    end

    if sum(Lxu_vec) ~= Ltot
        error('sum(Lxu)=%d must equal size(H,2)=%d', sum(Lxu_vec), Ltot);
    end

    % handle real baseband and single tone scaling
    N = N_in;
    single_tone_flag = (N_in == 1);
    scale_single = (3 - cb);

    if N_in > 1 && cb == 2

        if mod(N_in, 2) ~= 0
            error('for cb=2 and N>1, N must be even');
        end

        H = H(:, :, 1:(N_in / 2));
        N = N_in / 2;
    elseif single_tone_flag
        Eu = Eu / scale_single;
        H = reshape(H, Ly, Ltot, 1);
        N = 1;
    end

    % compute antenna index mapping
    idx_end = cumsum(Lxu_vec);
    idx_start = [1, idx_end(1:end - 1) + 1];

    %% ellipsoid method for dual optimization
    err_tol = 1e-9;
    max_iter = 2000;

    % initialize ellipsoid
    [A, w] = init_ellipsoid(H, Eu, theta, Lxu_vec, idx_start, idx_end);
    state_warm = cell(N, 1);

    iter = 0;

    while true
        iter = iter + 1;

        % evaluate Lagrangian at current w
        [~, bun_nats, Eun_iter, Rxx_iter, state_warm] = eval_lagfunc(H, theta, w, Lxu_vec, idx_start, idx_end, state_warm);

        % compute subgradient: g = Eu - sum(Eun, 2)
        g = Eu - sum(Eun_iter, 2);

        % check stopping criteria
        gAg = g' * A * g;

        if sqrt(gAg) <= err_tol
            break;
        end

        if iter > max_iter
            break;
        end

        % update ellipsoid
        gt = g / sqrt(gAg);
        w = w - 1 / (U + 1) * A * gt;
        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * A * gt * gt' * A);

        % project w onto feasible set (w >= 0)
        ind = find(w < 0);

        while ~isempty(ind)
            g_proj = zeros(U, 1);
            g_proj(ind(1)) = -1;
            gAg_proj = g_proj' * A * g_proj;
            gt_proj = g_proj / sqrt(gAg_proj);
            w = w - 1 / (U + 1) * A * gt_proj;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * A * gt_proj * gt_proj' * A);
            ind = find(w < 0);
        end

    end

    % final evaluation at converged w
    [~, bun_nats, Eun, Rxx_cell, ~] = eval_lagfunc(H, theta, w, Lxu_vec, idx_start, idx_end, state_warm);

    % scale covariances to exactly satisfy energy constraints
    E_used = sum(Eun, 2);

    for u = 1:U

        if E_used(u) > 1e-12
            % normal case: scale existing allocation
            scale_u = Eu(u) / E_used(u);

            for n = 1:N
                Rxx_cell{u, n} = scale_u * Rxx_cell{u, n};
                Eun(u, n) = scale_u * Eun(u, n);
            end

        elseif Eu(u) > 1e-12
            % edge case: user has no allocated energy but should have some
            % allocate energy uniformly across tones with identity covariance shape
            Lu = Lxu_vec(u);
            energy_per_tone = Eu(u) / N / Lu; % energy per antenna per tone

            for n = 1:N
                Rxx_cell{u, n} = energy_per_tone * eye(Lu);
                Eun(u, n) = Eu(u) / N;
            end

        end

    end

    % recompute rates with scaled covariances (using theta-sorted decoding order)
    bun_nats = compute_rates(H, Rxx_cell, theta, Lxu_vec, idx_start, idx_end, Ly, U, N);

    % match CVX convention for N==1 energy/covariance scaling
    if single_tone_flag
        Eun = scale_single * Eun;

        for u = 1:U
            Rxx_cell{u, 1} = scale_single * Rxx_cell{u, 1};
        end

        bun_nats = compute_rates(H, Rxx_cell, theta, Lxu_vec, idx_start, idx_end, Ly, U, N);
    end

    bun = bun_nats / (log(2) * cb);

    % assemble covariance outputs
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

end

function bun = compute_rates(H, Rxx_cell, theta, Lxu, idx_start, idx_end, Ly, U, N)
    %COMPUTE_RATES Compute per-user per-tone rates in nats
    %   Uses decoding order sorted by decreasing theta.

    % sort users by decreasing theta
    [~, order] = sort(theta, 'descend');

    bun = zeros(U, N);
    eye_Ly = eye(Ly);

    for n = 1:N
        Hn = H(:, :, n);
        S_prev = eye_Ly;
        logdet_prev = 0;

        for pos = 1:U
            u = order(pos);
            cols = idx_start(u):idx_end(u);
            Hu = Hn(:, cols);
            Ru = Rxx_cell{u, n};
            HRH = Hu * Ru * Hu';
            HRH = 0.5 * (HRH + HRH');
            S_curr = S_prev + HRH;
            S_curr = 0.5 * (S_curr + S_curr');

            logdet_curr = logdet(S_curr);
            bun(u, n) = logdet_curr - logdet_prev;
            S_prev = S_curr;
            logdet_prev = logdet_curr;
        end

    end

end

function val = logdet(M)
    %LOGDET Numerically stable log-determinant

    M = 0.5 * (M + M');
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-12 * eye(size(M, 1));
        [L, flag] = chol(M, 'lower');
    end

    if flag ~= 0
        eigvals = real(eig(M));
        eigvals = max(eigvals, 1e-15);
        val = sum(log(eigvals));
        return;
    end

    val = 2 * sum(log(diag(L)));

end
