function [f, b, Rxx, state_out] = solve_dual_tone(H, theta, w, Lx, idx_start, idx_end, R_init, state_in)
    %SOLVE_DUAL_TONE Dual maximization on a single tone for MIMO MAC
    %   [f, b, Rxx, state_out] = SOLVE_DUAL_TONE(H, theta, w, Lx, idx_start, idx_end, ...)
    %   solves the per-tone Lagrangian maximization that appears inside the
    %   ellipsoid method for the MIMO minPMAC problem. This implementation avoids
    %   CVX and works directly with the covariance matrices for each user by
    %   applying a limited memory quasi-newton ascent on the dual objective.
    %
    %   Inputs:
    %       H           Ly-by-Ltot channel matrix for this tone (users stacked).
    %       theta       U-by-1 dual variables (not yet sorted).
    %       w           U-by-1 user energy weights.
    %       Lx          U-by-1 number of antennas per user.
    %       idx_start   U-by-1 starting indices for each user's antennas.
    %       idx_end     U-by-1 ending indices for each user's antennas.
    %       R_init      (optional) Ltot-by-Ltot warm-start covariance matrix.
    %       state_in    (optional) structure with lbfgs history for warm-start.
    %
    %   Outputs:
    %       f         scalar value theta' * b - sum_u w_u * trace(R_u).
    %       b         U-by-1 rate vector in nats/channel use (consistent with SISO).
    %       Rxx       Ltot-by-Ltot transmit covariance matrix for this tone.
    %       state_out optional structure retaining lbfgs warm-start data.
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, ~] = size(H);
    U = length(Lx);

    % sort users according to theta (descending) to match polymatroid vertex
    [stheta, order] = sort(-theta);
    stheta = -stheta;
    sw = w(order);

    % build per-user channel blocks for the sorted order
    H_blocks = cell(U, 1);
    block_sizes = Lx(order);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        H_blocks{u} = H(:, cols);
    end

    % coefficients alpha_k = 0.5 * (theta_k - theta_{k+1}), theta_{U+1}=0
    alpha = 0.5 * (stheta - [stheta(2:U); 0]);

    if nargin < 8
        state_in = [];
    end

    state_out = [];
    state_valid = true;

    if ~isempty(state_in)

        if ~isfield(state_in, 'order') || ~isequal(state_in.order(:), order(:))
            state_in = [];
            state_valid = false;
        elseif ~isfield(state_in, 'block_sizes') || any(state_in.block_sizes(:) ~= block_sizes(:))
            state_in = [];
            state_valid = false;
        end

    else
        state_valid = false;
    end

    if ~state_valid

        if nargin < 7 || isempty(R_init)
            R_init_blocks = [];
        else
            R_init_blocks = cell(U, 1);

            for u = 1:U
                orig = order(u);
                cols = idx_start(orig):idx_end(orig);
                Ru = R_init(cols, cols);
                R_init_blocks{u} = 0.5 * (Ru + Ru');
            end

        end

    else
        R_init_blocks = [];
    end

    [R_blocks, ~, state_internal] = maximize_dual_lbfgs(H_blocks, alpha, sw, block_sizes, R_init_blocks, state_in);

    % compute achieved rates (nats) for sorted users
    b_sorted = zeros(U, 1);
    S_prev = eye(Ly);

    for u = 1:U
        S_curr = S_prev + H_blocks{u} * R_blocks{u} * H_blocks{u}';
        S_curr = 0.5 * (S_curr + S_curr');
        b_sorted(u) = 0.5 * real(logdet_spd(S_curr) - logdet_spd(S_prev));
        S_prev = S_curr;
    end

    % map rates back to original ordering
    b = zeros(U, 1);
    b(order) = b_sorted;

    % assemble full covariance matrix in original antenna order
    Ltot = sum(Lx);
    Rxx = zeros(Ltot, Ltot);

    for u = 1:U
        orig = order(u);
        cols = idx_start(orig):idx_end(orig);
        Ru = R_blocks{u};
        Rxx(cols, cols) = 0.5 * (Ru + Ru');
    end

    % objective value
    energy = 0;

    for u = 1:U
        orig = order(u);
        energy = energy + w(orig) * trace(R_blocks{u});
    end

    f = real(theta' * b - energy);

    if nargout >= 4
        state_out = finalize_state(state_internal, order, block_sizes, R_blocks);
    end

end

function [F, grad_phi_blocks, R_blocks] = eval_gradobj(H_blocks, B_blocks, alpha, w)
    % evaluate dual objective and gradient for given B blocks
    U = numel(H_blocks);
    Ly = size(H_blocks{1}, 1);
    eye_Ly = eye(Ly);
    S = eye_Ly;
    logdets = zeros(U, 1);
    invS_cache = cell(U, 1);
    energy_term = 0;

    for u = 1:U
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        HB = Hu * Bu;
        S = S + HB * HB';
        S = 0.5 * (S + S');
        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            [L, flag] = chol(S +1e-6 * eye_Ly, 'lower');
        end

        % if still failing, signal failure to backtracking line search
        if flag ~= 0
            F = -Inf;
            grad_phi_blocks = [];
            R_blocks = [];
            return;
        end

        logdets(u) = 2 * sum(log(diag(L)));
        Linv = L \ eye_Ly;
        invS = Linv' * Linv;
        invS_cache{u} = 0.5 * (invS + invS');
        energy_term = energy_term + w(u) * sum(real(conj(Bu(:)) .* Bu(:)));
    end

    F = sum(alpha .* logdets) - energy_term;
    grad_phi_blocks = cell(U, 1);
    R_blocks = cell(U, 1);
    tail = zeros(Ly, Ly);

    for u = U:-1:1
        tail = tail + alpha(u) * invS_cache{u};
        Hu = H_blocks{u};
        Bu = B_blocks{u};
        Lu = size(Bu, 1);
        grad_R = Hu' * (tail * Hu) - w(u) * eye(Lu);
        grad_R = 0.5 * (grad_R + grad_R');
        grad_phi_blocks{u} = -2 * grad_R * Bu;
        Ru = Bu * Bu';
        R_blocks{u} = 0.5 * (Ru + Ru');
    end

end

function [R_blocks, F_opt, state_out] = maximize_dual_lbfgs(H_blocks, alpha, w, block_sizes, R_init_blocks, state_in)
    % lbfgs parameters
    max_it = 200;
    tol_grad = 1e-6;
    m_hist = 12;
    step_init = 1.0;
    step_min = 1e-12;
    beta = 0.5;
    armijo_c = 1e-4;
    max_linesearch = 24;
    U = numel(H_blocks);

    if nargin < 6
        state_in = [];
    end

    [state_valid, prepared_state] = prepare_lbfgs_state(state_in, block_sizes, U);

    if state_valid
        B_blocks = prepared_state.B_blocks;
        S_hist = prepared_state.S_hist;
        Y_hist = prepared_state.Y_hist;
    else
        B_blocks = cell(U, 1);

        for u = 1:U

            if ~isempty(R_init_blocks)
                Ru = R_init_blocks{u};
                Ru = 0.5 * (Ru + Ru'); % inline symmetrize
                B_blocks{u} = initialize_factor(Ru);
            else
                B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
            end

        end

        S_hist = cell(0, 1);
        Y_hist = cell(0, 1);
    end

    x = pack_complex_blocks(B_blocks, block_sizes);

    if state_valid
        [S_hist, Y_hist] = sanitize_history(S_hist, Y_hist, numel(x), m_hist);
    end

    [phi, F_opt, grad, B_blocks, R_blocks] = eval_lbfgs_state(x, H_blocks, alpha, w, block_sizes);

    if isinf(phi)

        for u = 1:U
            B_blocks{u} = sqrt(1e-6) * eye(block_sizes(u));
        end

        x = pack_complex_blocks(B_blocks, block_sizes);
        S_hist = cell(0, 1);
        Y_hist = cell(0, 1);
        [phi, F_opt, grad, B_blocks, R_blocks] = eval_lbfgs_state(x, H_blocks, alpha, w, block_sizes);
    end

    grad_norm = norm(grad);

    for iter = 1:max_it

        if grad_norm < tol_grad
            break;
        end

        direction = lbfgs_direction(grad, S_hist, Y_hist);
        dir_deriv = dot_real(grad, direction);

        if dir_deriv >= 0
            direction = -grad;
            dir_deriv = dot_real(grad, direction);
            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        step = step_init;
        accepted = false;
        x_candidate = [];
        phi_candidate = [];
        grad_candidate = [];
        B_candidate = [];
        R_candidate = [];

        for ls = 1:max_linesearch
            x_trial = x + step * direction;
            [phi_trial, F_trial, grad_trial, B_trial, R_trial] = eval_lbfgs_state(x_trial, H_blocks, alpha, w, block_sizes);

            if phi_trial <= phi + armijo_c * step * dir_deriv
                accepted = true;
                x_candidate = x_trial;
                phi_candidate = phi_trial;
                grad_candidate = grad_trial;
                B_candidate = B_trial;
                R_candidate = R_trial;
                F_opt = F_trial;
                break;
            end

            step = step * beta;

            if step < step_min
                break;
            end

        end

        if ~accepted
            direction = -grad;
            dir_deriv = dot_real(grad, direction);
            step = step_init;

            for ls = 1:max_linesearch
                x_trial = x + step * direction;
                [phi_trial, F_trial, grad_trial, B_trial, R_trial] = eval_lbfgs_state(x_trial, H_blocks, alpha, w, block_sizes);

                if phi_trial <= phi + armijo_c * step * dir_deriv
                    accepted = true;
                    x_candidate = x_trial;
                    phi_candidate = phi_trial;
                    grad_candidate = grad_trial;
                    B_candidate = B_trial;
                    R_candidate = R_trial;
                    F_opt = F_trial;
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

            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        s = x_candidate - x;
        y = grad_candidate - grad;

        if dot_real(s, y) > 1e-12

            if numel(S_hist) == m_hist
                S_hist = S_hist(2:end);
                Y_hist = Y_hist(2:end);
            end

            S_hist{end + 1} = s;
            Y_hist{end + 1} = y;

        else
            S_hist = cell(0, 1);
            Y_hist = cell(0, 1);
        end

        x = x_candidate;
        phi = phi_candidate;
        grad = grad_candidate;
        B_blocks = B_candidate;
        R_blocks = R_candidate;
        grad_norm = norm(grad);
    end

    state_out = struct();
    state_out.B_blocks = B_blocks;
    state_out.S_hist = S_hist;
    state_out.Y_hist = Y_hist;
    state_out.x = x;
    state_out.grad = grad;
    state_out.phi = phi;
    state_out.F = F_opt;
end

function [phi, F_val, grad_vec, B_blocks, R_blocks] = eval_lbfgs_state(x, H_blocks, alpha, w, block_sizes)
    B_blocks = unpack_complex_blocks(x, block_sizes);
    [F_val, grad_blocks, R_blocks] = eval_gradobj(H_blocks, B_blocks, alpha, w);

    if F_val == -Inf
        phi = Inf;
        grad_vec = [];
        return;
    end

    grad_vec = pack_complex_blocks(grad_blocks, block_sizes);
    phi = -F_val;
end

function direction = lbfgs_direction(grad, S_hist, Y_hist)

    if isempty(S_hist)
        direction = -grad;
        return;
    end

    k = numel(S_hist);
    alpha = zeros(k, 1);
    rho = zeros(k, 1);
    q = grad;

    for i = k:-1:1
        s = S_hist{i};
        y = Y_hist{i};
        denom = dot_real(y, s);

        if denom <= 1e-12
            rho(i) = 0;
            alpha(i) = 0;
            continue;
        end

        rho(i) = 1 / denom;
        alpha(i) = rho(i) * dot_real(s, q);
        q = q - alpha(i) * y;
    end

    sk = S_hist{end};
    yk = Y_hist{end};
    denom = dot_real(yk, yk);

    if denom <= 1e-12
        H0 = 1;
    else
        H0 = dot_real(sk, yk) / denom;
    end

    r = H0 * q;

    for i = 1:k
        y = Y_hist{i};
        s = S_hist{i};

        if rho(i) == 0
            beta = 0;
        else
            beta = rho(i) * dot_real(y, r);
        end

        r = r + s * (alpha(i) - beta);
    end

    direction = -r;
end

function vec = pack_complex_blocks(blocks, block_sizes)
    U = numel(blocks);
    total_len = 0;

    for u = 1:U
        total_len = total_len + 2 * block_sizes(u) * block_sizes(u);
    end

    vec = zeros(total_len, 1);
    offset = 0;

    for u = 1:U
        block = blocks{u};
        flat = block(:);
        len = numel(flat);
        vec(offset + (1:len)) = real(flat);
        vec(offset + len + (1:len)) = imag(flat);
        offset = offset + 2 * len;
    end

end

function blocks = unpack_complex_blocks(vec, block_sizes)
    U = numel(block_sizes);
    blocks = cell(U, 1);
    offset = 0;

    for u = 1:U
        len = block_sizes(u) * block_sizes(u);
        real_part = vec(offset + (1:len));
        imag_part = vec(offset + len + (1:len));
        block = reshape(real_part + 1j * imag_part, block_sizes(u), block_sizes(u));
        blocks{u} = block;
        offset = offset + 2 * len;
    end

end

function val = dot_real(a, b)
    val = real(sum(a .* b));
end

function [is_valid, state_out] = prepare_lbfgs_state(state_in, block_sizes, U)
    is_valid = false;
    state_out = struct();

    if isempty(state_in) || ~isstruct(state_in)
        return;
    end

    if ~isfield(state_in, 'B_blocks')
        return;
    end

    if isfield(state_in, 'block_sizes')

        if numel(state_in.block_sizes) ~= numel(block_sizes)
            return;
        end

        if any(state_in.block_sizes(:) ~= block_sizes(:))
            return;
        end

    end

    if numel(state_in.B_blocks) ~= U
        return;
    end

    for u = 1:U
        Bu = state_in.B_blocks{u};

        if isempty(Bu) || any(size(Bu) ~= block_sizes(u))
            return;
        end

    end

    state_out = state_in;

    if ~isfield(state_out, 'S_hist') || ~iscell(state_out.S_hist)
        state_out.S_hist = cell(0, 1);
    end

    if ~isfield(state_out, 'Y_hist') || ~iscell(state_out.Y_hist)
        state_out.Y_hist = cell(0, 1);
    end

    is_valid = true;
end

function [S_hist_out, Y_hist_out] = sanitize_history(S_hist_in, Y_hist_in, dim, m_hist)

    if isempty(S_hist_in) || isempty(Y_hist_in)
        S_hist_out = cell(0, 1);
        Y_hist_out = cell(0, 1);
        return;
    end

    k = min([numel(S_hist_in), numel(Y_hist_in), m_hist]);
    start_idx = max(1, numel(S_hist_in) - k + 1);
    S_hist_out = cell(0, 1);
    Y_hist_out = cell(0, 1);

    for idx = start_idx:numel(S_hist_in)
        s = S_hist_in{idx};
        y = Y_hist_in{idx};

        if numel(s) == dim && numel(y) == dim
            S_hist_out{end + 1, 1} = s;
            Y_hist_out{end + 1, 1} = y;
        end

    end

end

function state_out = finalize_state(state_internal, order, block_sizes, R_blocks)

    if nargin < 1 || isempty(state_internal) || ~isstruct(state_internal)
        state_internal = struct();
    end

    state_out = state_internal;
    state_out.order = order;
    state_out.block_sizes = block_sizes;
    state_out.R_blocks = R_blocks;
end

function B = initialize_factor(R)

    if isempty(R)
        B = zeros(size(R));
        return;
    end

    R = 0.5 * (R + R');
    [L, flag] = chol(R, 'lower');

    if flag == 0
        B = L;
        return;
    end

    % fallback using eigen decomposition for semi-definite cases
    [V, D] = eig(R, 'vector');
    d = real(D);
    d(d < 0) = 0;
    B = V * diag(sqrt(d));
end

function val = logdet_spd(M)
    M = 0.5 * (M + M');
    [L, flag] = chol(M, 'lower');

    if flag ~= 0
        M = M +1e-6 * eye(size(M, 1));
        [L, flag] = chol(M, 'lower');
    end

    if flag ~= 0
        val = -Inf;
        return;
    end

    val = 2 * sum(log(diag(L)));
end
