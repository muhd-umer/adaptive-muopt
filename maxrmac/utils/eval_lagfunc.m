function [f, bun, Eun, Rxx_all, state_all] = eval_lagfunc(H, theta, w, Lxu, idx_start, idx_end, state_init)
    %EVAL_LAGFUNC Evaluate Lagrangian for maxRMAC across all tones
    %   [f, bun, Eun, Rxx_all, state_all] = EVAL_LAGFUNC(H, theta, w, ...)
    %   evaluates the dual objective by optimizing each tone independently.
    %
    %   Inputs:
    %       H           channel tensor [Ly, Ltot, N]
    %       theta       U-by-1 rate weights
    %       w           U-by-1 energy weights (Lagrange multipliers)
    %       Lxu         1-by-U antennas per user
    %       idx_start   1-by-U starting antenna indices
    %       idx_end     1-by-U ending antenna indices
    %       state_init  (optional) cell array of per-tone warm-start states
    %
    %   Outputs:
    %       f           scalar dual objective value
    %       bun         U x N per-user per-tone rates in nats
    %       Eun         U x N per-user per-tone energies
    %       Rxx_all     U x N cell array of covariance matrices
    %       state_all   cell array of per-tone states for warm-start
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [~, ~, N] = size(H);
    U = length(Lxu);

    f = 0;
    bun = zeros(U, N);
    Eun = zeros(U, N);
    Rxx_all = cell(U, N);

    if nargin < 7 || isempty(state_init)
        state_init = cell(N, 1);
    end

    state_all = cell(N, 1);

    % optimize each tone
    for n = 1:N

        if numel(state_init) >= n
            state_n = state_init{n};
        else
            state_n = [];
        end

        [f_n, b_n, Rxx_n, E_n, state_out] = solve_tone(H(:, :, n), theta, w, Lxu, idx_start, idx_end, state_n);

        f = f + f_n;
        bun(:, n) = b_n;
        Eun(:, n) = E_n;

        for u = 1:U
            Rxx_all{u, n} = Rxx_n{u};
        end

        state_all{n} = state_out;
    end

end
