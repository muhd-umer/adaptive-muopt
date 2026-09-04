function [f, bun, Rxx_all, state_all] = eval_lagfunc(H, theta, w, bu_min, Lx, idx_start, idx_end, R_init_all, state_init_all)
    %EVAL_LAGFUNC Evaluate Lagrangian dual function for MIMO MAC
    %   [f, bun, Rxx_all, state_all] = EVAL_LAGFUNC(H, theta, w, bu_min, Lx, ...)
    %   evaluates the dual objective by optimizing each tone independently via
    %   solve_dual_tone, returning both the per-user rates in nats and the
    %   stacked covariance matrices.
    %
    %   Inputs:
    %       H             channel tensor [Ly, Ltot, N]
    %       theta         U-by-1 dual variables
    %       w             U-by-1 energy weights
    %       bu_min        U-by-1 target rates in nats
    %       Lx            1-by-U antennas per user
    %       idx_start     1-by-U starting antenna indices
    %       idx_end       1-by-U ending antenna indices
    %       R_init_all    (optional) Ltot x Ltot x N warm-start covariances
    %       state_init_all (optional) cell array of per-tone lbfgs states
    %
    %   Outputs:
    %       f             scalar dual objective value
    %       bun           U x N per-user per-tone rates in nats
    %       Rxx_all       Ltot x Ltot x N covariance matrices
    %       state_all     cell array of per-tone lbfgs states for warm-start
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    f = 0;
    bun = zeros(U, N);
    Rxx_all = zeros(Ltot, Ltot, N);

    if nargin < 8
        R_init_all = [];
    end

    if nargin < 9 || isempty(state_init_all)
        state_init_all = cell(0, 1);
    end

    states_cell = cell(N, 1);

    % optimize each tone
    for n = 1:N

        if isempty(R_init_all)
            R_init_n = [];
        else
            R_init_n = R_init_all(:, :, n);
        end

        if isempty(state_init_all) || numel(state_init_all) < n
            state_init_n = [];
        else
            state_init_n = state_init_all{n};
        end

        [temp_f, b_n, Rxx_n, state_n] = solve_dual_tone(H(:, :, n), theta, w, Lx, idx_start, idx_end, R_init_n, state_init_n);
        f = f + temp_f;
        bun(:, n) = b_n;
        Rxx_all(:, :, n) = Rxx_n;
        states_cell{n} = state_n;
    end

    f = f - theta' * bu_min;

    if nargout >= 4
        state_all = states_cell;
    end

end
