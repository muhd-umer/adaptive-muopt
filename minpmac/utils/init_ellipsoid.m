function [A, g, w] = init_ellipsoid(H, bu, w, cb, Lx, idx_start, idx_end)
    %INIT_ELLIPSOID Initialize ellipsoid method for MIMO MAC
    %   [A, g, w] = INIT_ELLIPSOID(H, bu, w, cb, Lx, idx_start, idx_end)
    %   returns the initial ellipsoid shape matrix and center for the minPMAC
    %   solver by approximating user energies via successive water-filling sweeps.
    %
    %   Inputs:
    %       H           channel tensor [Ly, Ltot, N]
    %       bu          1-by-U target rates
    %       w           1-by-U energy weights
    %       cb          baseband type (1=complex, 2=real)
    %       Lx          1-by-U antennas per user
    %       idx_start   1-by-U starting antenna indices
    %       idx_end     1-by-U ending antenna indices
    %
    %   Outputs:
    %       A           U x U initial ellipsoid shape matrix
    %       g           U x 1 initial ellipsoid center
    %       w           (pass-through) energy weights
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University

    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    order = 1:U;
    tmax = zeros(U, 1);

    % for each user, compute water-filling solution with perturbed rate
    for u = 1:U
        en = zeros(U, N);
        b = bu;
        b(u) = b(u) + 1; % perturb by 1 nat (as in reference)

        % iterate through users in reverse order
        for u1 = U:-1:1
            % initialize noise covariance for each tone
            Rnoise = zeros(Ly, Ly, N);

            for index = 1:N
                Rnoise(:, :, index) = eye(Ly);
            end

            % add interference from other users
            for u2 = U:-1:1

                if u2 ~= u1

                    for index = 1:N
                        ant_idx = idx_start(u2):idx_end(u2);
                        H_u2 = H(:, ant_idx, index);

                        % for siso, en(u2, index) is scalar energy
                        if Lx(u2) == 1
                            Rnoise(:, :, index) = Rnoise(:, :, index) + ...
                                en(u2, index) * H_u2 * H_u2';
                        else
                            % for mimo, use scalar approximation
                            energy_per_ant = en(u2, index) / Lx(u2);
                            Rnoise(:, :, index) = Rnoise(:, :, index) + ...
                                energy_per_ant * (H_u2 * H_u2');
                        end

                    end

                end

            end

            % compute effective channel gains per tone
            ant_idx = idx_start(u1):idx_end(u1);
            g_tones = zeros(1, N);

            for index = 1:N
                H_u1 = H(:, ant_idx, index);
                h_temp = sqrtm(Rnoise(:, :, index)) \ H_u1;

                if Lx(u1) == 1
                    % siso case: use squared norm
                    g_tones(index) = (h_temp' * h_temp);
                else
                    % mimo case: use maximum singular value squared
                    sv = svd(h_temp);
                    g_tones(index) = sv(1) ^ 2;
                end

            end

            % water-filling allocation
            [b_temp, E_temp] = fmwaterfill_gn(g_tones, b(u1) / N, 0, cb);
            en(u1, :) = E_temp;
        end

        % compute weighted energy sum
        tmax(u) = sum(w .* sum(en, 2));
    end

    % initialize ellipsoid center and shape
    g = tmax / 2;
    A = eye(U) * sum(g .^ 2);
end
