function [A, w_init] = init_ellipsoid(H, Eu, theta, Lxu, idx_start, idx_end)
    %INIT_ELLIPSOID Initialize ellipsoid for maxRMAC dual optimization
    %   [A, w_init] = INIT_ELLIPSOID(H, Eu, theta, Lxu, idx_start, idx_end)
    %   computes the initial ellipsoid shape matrix and center for the Lagrange
    %   multipliers w in the rate maximization problem.
    %
    %   Inputs:
    %       H           channel tensor [Ly, Ltot, N]
    %       Eu          U-by-1 per-user energy constraints
    %       theta       U-by-1 rate weights
    %       Lxu         1-by-U antennas per user
    %       idx_start   1-by-U starting antenna indices
    %       idx_end     1-by-U ending antenna indices
    %
    %   Outputs:
    %       A           U x U initial ellipsoid shape matrix
    %       w_init      U x 1 initial ellipsoid center (dual variables)
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, ~, N] = size(H);
    U = length(Lxu);
    Eu = Eu(:);
    theta = theta(:);

    % compute approximate upper bounds on w
    w_max = zeros(U, 1);

    for u = 1:U
        total_gain = 0;

        for n = 1:N
            cols = idx_start(u):idx_end(u);
            Hu = H(:, cols, n);

            if Lxu(u) == 1
                gain = real(Hu' * Hu);
            else
                sv = svd(Hu);
                gain = sum(sv .^ 2);
            end

            total_gain = total_gain + gain;
        end

        % approximate w as gradient of rate at energy boundary
        if Eu(u) > 1e-12 && total_gain > 0
            avg_gain = total_gain / N;
            % w ~ theta * d/dE log(1 + g*E) = theta * g / (1 + g*E)
            w_max(u) = theta(u) * avg_gain / (1 + avg_gain * Eu(u) / N);
        else
            w_max(u) = theta(u);
        end

    end

    % ensure minimum value and expand bounds
    w_max = max(w_max, 0.1);

    % initialize ellipsoid (sphere centered at w_max/2)
    w_init = w_max / 2;
    radius_sq = sum(w_max .^ 2) / 4;
    A = eye(U) * max(radius_sq, 1);

end
