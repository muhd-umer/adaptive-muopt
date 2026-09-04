function [f, b, Rxx, E, state_out] = solve_tone(H, theta, w, Lxu, idx_start, idx_end, state_in)
    %SOLVE_TONE Per-tone Lagrangian optimization for maxRMAC
    %   [f, b, Rxx, E, state_out] = SOLVE_TONE(H, theta, w, Lxu, ...)
    %   solves the per-tone optimization for rate maximization subject to
    %   weighted energy penalty. Uses L-BFGS over Cholesky factors.
    %
    %   Maximizes: sum_u delta(u) * log|S_u| - sum_u w(u) * trace(Rxx_u)
    %   where S_u = I + sum_{j=1}^u H_j * Rxx_j * H_j' and delta = theta - theta_{next}
    %
    %   Inputs:
    %       H           Ly-by-Ltot channel matrix for this tone
    %       theta       U-by-1 rate weights
    %       w           U-by-1 energy weights (Lagrange multipliers)
    %       Lxu         1-by-U antennas per user
    %       idx_start   1-by-U starting antenna indices
    %       idx_end     1-by-U ending antenna indices
    %       state_in    (optional) warm-start state from previous solve
    %
    %   Outputs:
    %       f           scalar Lagrangian value
    %       b           U-by-1 per-user rates in nats
    %       Rxx         U-by-1 cell array of covariance matrices
    %       E           U-by-1 per-user energies
    %       state_out   state structure for warm-starting
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, ~] = size(H);
    U = length(Lxu);

    % sort users by decreasing theta for polymatroid vertex
    [stheta, order] = sort(theta, 'descend');
    sw = w(order);
    sLxu = Lxu(order);

    % build per-user channel blocks in sorted order
    H_blocks = cell(U, 1);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        H_blocks{u} = H(:, cols);
    end

    % coefficients: delta_k = theta_k - theta_{k+1}, theta_{U+1} = 0
    delta = stheta - [stheta(2:U); 0];

    % process warm-start state
    state_valid = false;

    if nargin >= 7 && ~isempty(state_in)

        if isstruct(state_in) && isfield(state_in, 'order') && isequal(state_in.order(:), order(:))

            if isfield(state_in, 'block_sizes') && isequal(state_in.block_sizes(:), sLxu(:))
                state_valid = true;
            end

        end

    end

    if ~state_valid
        state_in = [];
    end

    % run l-bfgs optimization
    [R_blocks, state_internal] = maximize_lagrangian_lbfgs(H_blocks, delta, sw, sLxu, Ly, state_in);

    % compute achieved rates (nats) for sorted users
    b_sorted = zeros(U, 1);
    S_prev = eye(Ly);

    for u = 1:U
        S_curr = S_prev + H_blocks{u} * R_blocks{u} * H_blocks{u}';
        S_curr = 0.5 * (S_curr + S_curr');
        b_sorted(u) = real(logdet_spd(S_curr) - logdet_spd(S_prev));
        S_prev = S_curr;
    end

    % compute energies for sorted users
    E_sorted = zeros(U, 1);

    for u = 1:U
        E_sorted(u) = real(trace(R_blocks{u}));
    end

    % map back to original ordering
    b = zeros(U, 1);
    E = zeros(U, 1);
    Rxx = cell(U, 1);
    b(order) = b_sorted;
    E(order) = E_sorted;

    for u = 1:U
        orig = order(u);
        Rxx{orig} = R_blocks{u};
    end

    % compute Lagrangian value
    f = theta' * b - w' * E;

    % prepare output state
    if nargout >= 5
        state_out = state_internal;
        state_out.order = order;
        state_out.block_sizes = sLxu;
    else
        state_out = [];
    end

end

function [R_blocks, state_out] = maximize_lagrangian_lbfgs(H_blocks, delta, w, block_sizes, Ly, state_in)
    %MAXIMIZE_LAGRANGIAN_LBFGS L-BFGS solver for per-tone Lagrangian

    % l-bfgs parameters
    max_iter = 200;
    tol_grad = 1e-6;
    m_hist = 12;
    step_init = 1.0;
    step_min = 1e-12;
    beta = 0.5;
    armijo_c = 1e-4;
    max_linesearch = 24;

    U = numel(H_blocks);

    % initialize cholesky factors
    B_blocks = cell(U, 1);
    S_hist = {};
    Y_hist = {};

    if ~isempty(state_in) && isfield(state_in, 'B_blocks')
        valid = true;

        for u = 1:U
            Bu = state_in.B_blocks{u};

            if isempty(Bu) || any(size(Bu) ~= block_sizes(u))
                valid = false;
                break;
            end

        end

        if valid
            B_blocks = state_in.B_blocks;

            if isfield(state_in, 'S_hist')
                S_hist = state_in.S_hist;
            end

            if isfield(state_in, 'Y_hist')
                Y_hist = state_in.Y_hist;
            end

        else

            for u = 1:U
                B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
            end

        end

    else

        for u = 1:U
            B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
        end

    end

    x = pack_blocks(B_blocks, block_sizes);
    [phi, grad, R_blocks] = eval_objective(x, H_blocks, delta, w, block_sizes, Ly);

    if isinf(phi)
        % reset to safe initial point

        for u = 1:U
            B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
        end

        x = pack_blocks(B_blocks, block_sizes);
        S_hist = {};
        Y_hist = {};
        [phi, grad, R_blocks] = eval_objective(x, H_blocks, delta, w, block_sizes, Ly);
    end

    % main l-bfgs loop
    for iter = 1:max_iter

        if norm(grad) < tol_grad
            break;
        end

        dir = lbfgs_direction(grad, S_hist, Y_hist);
        dir_deriv = real(grad' * dir);

        if dir_deriv >= 0
            dir = -grad;
            dir_deriv = real(grad' * dir);
            S_hist = {};
            Y_hist = {};
        end

        step = step_init;
        accepted = false;
        x_new = x;
        phi_new = phi;
        grad_new = grad;
        R_new = R_blocks;

        for ls = 1:max_linesearch
            x_trial = x + step * dir;
            [phi_trial, grad_trial, R_trial] = eval_objective(x_trial, H_blocks, delta, w, block_sizes, Ly);

            if ~isinf(phi_trial) && phi_trial <= phi + armijo_c * step * dir_deriv
                accepted = true;
                x_new = x_trial;
                phi_new = phi_trial;
                grad_new = grad_trial;
                R_new = R_trial;
                break;
            end

            step = step * beta;

            if step < step_min
                break;
            end

        end

        if ~accepted
            % fall back to gradient descent
            dir = -grad;
            dir_deriv = real(grad' * dir);
            step = step_init;

            for ls = 1:max_linesearch
                x_trial = x + step * dir;
                [phi_trial, grad_trial, R_trial] = eval_objective(x_trial, H_blocks, delta, w, block_sizes, Ly);

                if ~isinf(phi_trial) && phi_trial <= phi + armijo_c * step * dir_deriv
                    accepted = true;
                    x_new = x_trial;
                    phi_new = phi_trial;
                    grad_new = grad_trial;
                    R_new = R_trial;
                    break;
                end

                step = step * beta;

                if step < step_min
                    break;
                end

            end

            if ~accepted
                break;
            end

            S_hist = {};
            Y_hist = {};
        end

        s = x_new - x;
        y = grad_new - grad;
        sy = real(s' * y);

        if sy > 1e-12

            if length(S_hist) >= m_hist
                S_hist = S_hist(2:end);
                Y_hist = Y_hist(2:end);
            end

            S_hist{end + 1} = s;
            Y_hist{end + 1} = y;

        else
            S_hist = {};
            Y_hist = {};
        end

        x = x_new;
        phi = phi_new;
        grad = grad_new;
        R_blocks = R_new;
    end

    % extract final R blocks from B blocks
    B_blocks = unpack_blocks(x, block_sizes);

    for u = 1:U
        Ru = B_blocks{u} * B_blocks{u}';
        R_blocks{u} = 0.5 * (Ru + Ru');
    end

    state_out = struct();
    state_out.B_blocks = B_blocks;
    state_out.S_hist = S_hist;
    state_out.Y_hist = Y_hist;

end

function [phi, grad_vec, R_blocks] = eval_objective(x, H_blocks, delta, w, block_sizes, Ly)
    %EVAL_OBJECTIVE Evaluate objective and gradient

    B_blocks = unpack_blocks(x, block_sizes);
    U = numel(H_blocks);
    eye_Ly = eye(Ly);
    S = eye_Ly;
    logdet_S = zeros(U, 1);
    invS = cell(U, 1);
    R_blocks = cell(U, 1);
    energy_term = 0;

    % forward pass: compute S_u and log|S_u|
    for u = 1:U
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        Ru = Bu * Bu';
        R_blocks{u} = 0.5 * (Ru + Ru');

        HRH = Hu * Ru * Hu';
        S = S + HRH;
        S = 0.5 * (S + S');

        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            [L, flag] = chol(S +1e-9 * eye_Ly, 'lower');
        end

        if flag ~= 0
            phi = Inf;
            grad_vec = [];
            R_blocks = [];
            return;
        end

        logdet_S(u) = 2 * sum(log(diag(L)));
        Linv = L \ eye_Ly;
        invS{u} = Linv' * Linv;
        energy_term = energy_term + w(u) * sum(real(conj(Bu(:)) .* Bu(:)));
    end

    F = sum(delta .* logdet_S) - real(energy_term);
    phi = -F; % minimize negative

    % backward pass: compute gradients
    grad_blocks = cell(U, 1);
    tail = zeros(Ly, Ly);

    for u = U:-1:1
        tail = tail + delta(u) * invS{u};
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        Lu = size(Bu, 1);

        % gradient of F w.r.t. R_u
        grad_R = Hu' * tail * Hu - w(u) * eye(Lu);
        grad_R = 0.5 * (grad_R + grad_R');

        % gradient of F w.r.t. B_u (chain rule)
        grad_B = 2 * grad_R * Bu;

        % negate for minimization
        grad_blocks{u} = -grad_B;
    end

    grad_vec = pack_blocks(grad_blocks, block_sizes);

end

function dir = lbfgs_direction(grad, S_hist, Y_hist)
    %LBFGS_DIRECTION Compute L-BFGS search direction

    if isempty(S_hist)
        dir = -grad;
        return;
    end

    k = length(S_hist);
    alpha = zeros(k, 1);
    rho = zeros(k, 1);
    q = grad;

    for i = k:-1:1
        s = S_hist{i};
        y = Y_hist{i};
        denom = real(y' * s);

        if denom <= 1e-12
            rho(i) = 0;
            alpha(i) = 0;
            continue;
        end

        rho(i) = 1 / denom;
        alpha(i) = rho(i) * real(s' * q);
        q = q - alpha(i) * y;
    end

    sk = S_hist{end};
    yk = Y_hist{end};
    denom = real(yk' * yk);

    if denom <= 1e-12
        H0 = 1;
    else
        H0 = real(sk' * yk) / denom;
    end

    r = H0 * q;

    for i = 1:k

        if rho(i) == 0
            continue;
        end

        y = Y_hist{i};
        s = S_hist{i};
        beta_val = rho(i) * real(y' * r);
        r = r + s * (alpha(i) - beta_val);
    end

    dir = -r;

end

function vec = pack_blocks(blocks, block_sizes)
    %PACK_BLOCKS Pack cell array of matrices into a vector

    U = length(block_sizes);
    total = 0;

    for u = 1:U
        total = total + 2 * block_sizes(u) ^ 2;
    end

    vec = zeros(total, 1);
    offset = 0;

    for u = 1:U
        B = blocks{u};
        len = numel(B);
        vec(offset + (1:len)) = real(B(:));
        vec(offset + len + (1:len)) = imag(B(:));
        offset = offset + 2 * len;
    end

end

function blocks = unpack_blocks(vec, block_sizes)
    %UNPACK_BLOCKS Unpack vector into cell array of matrices

    U = length(block_sizes);
    blocks = cell(U, 1);
    offset = 0;

    for u = 1:U
        Lu = block_sizes(u);
        len = Lu ^ 2;
        re = vec(offset + (1:len));
        im = vec(offset + len + (1:len));
        blocks{u} = reshape(re + 1j * im, Lu, Lu);
        offset = offset + 2 * len;
    end

end

function val = logdet_spd(M)
    %LOGDET_SPD Stable log-determinant for SPD matrix

    M = 0.5 * (M + M');
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-9 * eye(size(M, 1));
        [L, flag] = chol(M, 'lower');
    end

    if flag ~= 0
        val = -Inf;
        return;
    end

    val = 2 * sum(log(diag(L)));

end
