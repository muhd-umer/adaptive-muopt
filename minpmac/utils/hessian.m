function [g, H] = hessian(theta, Hmat, Rxx, w, Lx, idx_start, idx_end, ind, Ly)
    %HESSIAN Compute gradient and Hessian contributions for MIMO users
    %   [g, H] = HESSIAN(theta, Hmat, Rxx, w, Lx, idx_start, idx_end, ind, Ly)
    %   computes the gradient and Hessian of the Lagrangian with respect to
    %   the transmit covariance matrices for the min-power MAC problem.
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    U = length(Lx);
    Ltot = sum(Lx);
    theta_diff = 0.5 * (theta - [theta(2:U); 0]);

    % compute RYY inverses recursively using matrix inversion lemma
    RYY_inv = zeros(Ly, Ly, U);
    RYY_inv(:, :, 1) = eye(Ly);

    for u = 1:U
        ant_idx = idx_start(ind(u)):idx_end(ind(u));
        H_u = Hmat(:, ant_idx);
        R_u = Rxx(ant_idx, ant_idx);

        if u == 1
            temp = eye(Lx(ind(u))) + H_u' * H_u * R_u;
            RYY_inv(:, :, u) = eye(Ly) - H_u * (temp \ (R_u * H_u'));
        else
            temp = eye(Lx(ind(u))) + H_u' * RYY_inv(:, :, u - 1) * H_u * R_u;
            RYY_inv(:, :, u) = RYY_inv(:, :, u - 1) - ...
                RYY_inv(:, :, u - 1) * H_u * (temp \ (R_u * H_u' * RYY_inv(:, :, u - 1)));
        end

    end

    % compute gradient (vectorized)
    g = zeros(Ltot ^ 2, 1);

    for u = 1:U
        ant_idx = idx_start(ind(u)):idx_end(ind(u));
        H_u = Hmat(:, ant_idx);
        R_u = Rxx(ant_idx, ant_idx);

        grad_u = zeros(Lx(ind(u)), Lx(ind(u)));

        % sum over j >= u
        for j = u:U
            grad_u = grad_u + theta_diff(j) * (H_u' * RYY_inv(:, :, j) * H_u);
        end

        % barrier gradient
        try
            R_inv = inv(R_u + eye(Lx(ind(u))) * 1e-8);
            grad_u = grad_u + R_inv;
        catch
            R_inv = pinv(R_u + eye(Lx(ind(u))) * 1e-8);
            grad_u = grad_u + R_inv;
        end

        grad_u = grad_u - w(u) * eye(Lx(ind(u)));

        % place in vectorized gradient
        g_indices = [];

        for i = 1:length(ant_idx)

            for j = 1:length(ant_idx)
                idx = (ant_idx(j) - 1) * Ltot + ant_idx(i);
                g_indices = [g_indices; idx];
            end

        end

        g(g_indices) = grad_u(:);
    end

    % compute hessian (approximate with diagonal blocks for efficiency)
    H = zeros(Ltot ^ 2, Ltot ^ 2);

    for u = 1:U
        ant_idx = idx_start(ind(u)):idx_end(ind(u));
        H_u = Hmat(:, ant_idx);
        R_u = Rxx(ant_idx, ant_idx);
        L_u = Lx(ind(u));

        hess_u = zeros(L_u * L_u, L_u * L_u);

        % diagonal approximation for computational efficiency
        for j = u:U
            M = RYY_inv(:, :, j);
            term = -theta_diff(j) * kron((H_u' * M * H_u)', (H_u' * M * H_u));
            hess_u = hess_u + term;
        end

        % barrier hessian
        try
            R_inv = inv(R_u + eye(L_u) * 1e-8);
            barrier_hess = -kron(R_inv', R_inv);
            hess_u = hess_u + barrier_hess;
        catch
            R_inv = pinv(R_u + eye(L_u) * 1e-8);
            barrier_hess = -kron(R_inv', R_inv);
            hess_u = hess_u + barrier_hess;
        end

        % place in block-diagonal hessian
        h_indices = [];

        for i = 1:length(ant_idx)

            for j = 1:length(ant_idx)
                idx = (ant_idx(j) - 1) * Ltot + ant_idx(i);
                h_indices = [h_indices; idx];
            end

        end

        H(h_indices, h_indices) = hess_u;
    end

    % ensure hessian is symmetric and well-conditioned
    H = (H + H') / 2;
    H = H + eye(size(H)) * max(abs(diag(H))) * 1e-6;
end
