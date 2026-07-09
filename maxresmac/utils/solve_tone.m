function [Rxx_opt, E_opt] = solve_tone(Hn, Lxu, idx_start, idx_end, delta, lambda, U, Ly)
    %SOLVE_TONE Solve single-tone optimization using L-BFGS
    %   [Rxx_opt, E_opt] = SOLVE_TONE(Hn, Lxu, idx_start, idx_end, delta, lambda, U, Ly)
    %   solves the per-tone rate maximization problem using L-BFGS optimization
    %   over Cholesky factors for automatic PSD constraint satisfaction.
    %
    %   Maximizes: sum_u delta(u) * log|S_u| - lambda * sum_u trace(Rxx_u)
    %   where S_u = I + sum_{j=1}^u H_j * Rxx_j * H_j'
    %
    %   Inputs:
    %       Hn          Ly-by-Ltot channel matrix for single tone
    %       Lxu         1-by-U antennas per user (sorted order)
    %       idx_start   1-by-U starting antenna indices (sorted order)
    %       idx_end     1-by-U ending antenna indices (sorted order)
    %       delta       U-by-1 rate weight differences
    %       lambda      scalar Lagrange multiplier for energy constraint
    %       U           number of users
    %       Ly          number of receiver antennas
    %
    %   Outputs:
    %       Rxx_opt     U-by-1 cell array of optimal covariance matrices
    %       E_opt       U-by-1 vector of per-user energies

    % initialize B factors
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University
    B_blocks = cell(U, 1);

    for u = 1:U
        B_blocks{u} = sqrt(1e-6) * eye(Lxu(u));
    end

    % l-bfgs parameters
    max_iter = 200;
    tol_grad = 1e-7;
    m_hist = 10;
    step_init = 1.0;
    step_min = 1e-14;
    beta = 0.5;
    armijo_c = 1e-4;
    max_ls = 30;

    S_hist = {};
    Y_hist = {};

    % initial evaluation
    [phi, grad_blocks, R_blocks] = eval_gradobj(B_blocks, Hn, delta, lambda, Lxu, idx_start, idx_end, U, Ly);
    grad = pack_blocks(grad_blocks, Lxu);
    x = pack_blocks(B_blocks, Lxu);

    for iter = 1:max_iter
        grad_norm = norm(grad);

        if grad_norm < tol_grad
            break;
        end

        % compute search direction
        dir = lbfgs_direction(grad, S_hist, Y_hist);
        dir_deriv = real(grad' * dir);

        if dir_deriv >= 0
            dir = -grad;
            dir_deriv = real(grad' * dir);
            S_hist = {};
            Y_hist = {};
        end

        % backtracking line search
        step = step_init;
        accepted = false;

        for ls = 1:max_ls
            x_new = x + step * dir;
            B_new = unpack_blocks(x_new, Lxu);
            [phi_new, grad_new_blocks, R_new] = eval_gradobj(B_new, Hn, delta, lambda, Lxu, idx_start, idx_end, U, Ly);

            if phi_new <= phi + armijo_c * step * dir_deriv
                accepted = true;
                break;
            end

            step = step * beta;

            if step < step_min
                break;
            end

        end

        if ~accepted
            % try gradient descent
            dir = -grad;
            dir_deriv = real(grad' * dir);
            step = step_init;

            for ls = 1:max_ls
                x_new = x + step * dir;
                B_new = unpack_blocks(x_new, Lxu);
                [phi_new, grad_new_blocks, R_new] = eval_gradobj(B_new, Hn, delta, lambda, Lxu, idx_start, idx_end, U, Ly);

                if phi_new <= phi + armijo_c * step * dir_deriv
                    accepted = true;
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

        % update history
        s = x_new - x;
        grad_new = pack_blocks(grad_new_blocks, Lxu);
        y = grad_new - grad;
        sy = real(s' * y);

        if sy > 1e-12
            S_hist{end + 1} = s;
            Y_hist{end + 1} = y;

            if length(S_hist) > m_hist
                S_hist = S_hist(2:end);
                Y_hist = Y_hist(2:end);
            end

        else
            S_hist = {};
            Y_hist = {};
        end

        x = x_new;
        phi = phi_new;
        grad = grad_new;
        B_blocks = B_new;
        R_blocks = R_new;
    end

    % extract optimal Rxx
    Rxx_opt = cell(U, 1);
    E_opt = zeros(U, 1);

    for u = 1:U
        Ru = B_blocks{u} * B_blocks{u}';
        Rxx_opt{u} = 0.5 * (Ru + Ru');
        E_opt(u) = real(trace(Rxx_opt{u}));
    end

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
        beta = rho(i) * real(y' * r);
        r = r + s * (alpha(i) - beta);
    end

    dir = -r;
end

function vec = pack_blocks(blocks, Lxu)
    %PACK_BLOCKS Pack cell array of matrices into a vector

    U = length(blocks);
    total = 0;

    for u = 1:U
        total = total + 2 * Lxu(u) ^ 2;
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

function blocks = unpack_blocks(vec, Lxu)
    %UNPACK_BLOCKS Unpack vector into cell array of matrices

    U = length(Lxu);
    blocks = cell(U, 1);
    offset = 0;

    for u = 1:U
        Lu = Lxu(u);
        len = Lu ^ 2;
        re = vec(offset + (1:len));
        im = vec(offset + len + (1:len));
        blocks{u} = reshape(re + 1j * im, Lu, Lu);
        offset = offset + 2 * len;
    end

end
