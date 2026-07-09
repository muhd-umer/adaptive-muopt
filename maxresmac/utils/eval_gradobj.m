function [phi, grad_blocks, R_blocks] = eval_gradobj(B_blocks, Hn, delta, lambda, Lxu, idx_start, idx_end, U, Ly)
    %EVAL_GRADOBJ Evaluate objective and gradient for L-BFGS optimization
    %   [phi, grad_blocks, R_blocks] = EVAL_GRADOBJ(B_blocks, Hn, delta, lambda, ...)
    %   evaluates the negative dual objective (for minimization) and computes
    %   gradients with respect to the Cholesky factors B.
    %
    %   Inputs:
    %       B_blocks    U-by-1 cell array of Cholesky factors (Rxx = B*B')
    %       Hn          Ly-by-Ltot channel matrix for single tone
    %       delta       U-by-1 rate weight differences
    %       lambda      scalar Lagrange multiplier for energy constraint
    %       Lxu         1-by-U antennas per user (sorted order)
    %       idx_start   1-by-U starting antenna indices (sorted order)
    %       idx_end     1-by-U ending antenna indices (sorted order)
    %       U           number of users
    %       Ly          number of receiver antennas
    %
    %   Outputs:
    %       phi         scalar negative objective (for minimization)
    %       grad_blocks U-by-1 cell array of gradients w.r.t. B_u
    %       R_blocks    U-by-1 cell array of covariance matrices Rxx_u
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    eye_Ly = eye(Ly);
    S = eye_Ly;
    logdet_S = zeros(U, 1);
    invS = cell(U, 1);
    R_blocks = cell(U, 1);

    % forward pass: compute S_u and log|S_u|
    for u = 1:U
        cols = idx_start(u):idx_end(u);
        Hu = Hn(:, cols);
        Bu = B_blocks{u};
        Ru = Bu * Bu';
        R_blocks{u} = Ru;

        HRH = Hu * Ru * Hu';
        S = S + HRH;
        S = 0.5 * (S + S');

        [L, flag] = chol(S, 'lower');

        if flag ~= 0
            S = S +1e-9 * eye_Ly;
            [L, flag] = chol(S, 'lower');

            if flag ~= 0
                phi = Inf;
                grad_blocks = [];
                return;
            end

        end

        logdet_S(u) = 2 * sum(log(diag(L)));
        Linv = L \ eye_Ly;
        invS{u} = Linv' * Linv;
    end

    % compute objective
    energy = 0;

    for u = 1:U
        energy = energy + trace(R_blocks{u});
    end

    F = sum(delta .* logdet_S) - lambda * real(energy);
    phi = -F; % minimize negative

    % backward pass: compute gradients
    grad_blocks = cell(U, 1);
    tail = zeros(Ly, Ly);

    for u = U:-1:1
        tail = tail + delta(u) * invS{u};
        cols = idx_start(u):idx_end(u);
        Hu = Hn(:, cols);
        Bu = B_blocks{u};
        Lu = Lxu(u);

        % gradient of F w.r.t. Rxx_u
        grad_Rxx = Hu' * tail * Hu - lambda * eye(Lu);
        grad_Rxx = 0.5 * (grad_Rxx + grad_Rxx');

        % gradient of F w.r.t. Bu (chain rule: d/dB = 2 * grad_R * B)
        grad_B = 2 * grad_Rxx * Bu;

        % negate for minimization
        grad_blocks{u} = -grad_B;
    end

end
