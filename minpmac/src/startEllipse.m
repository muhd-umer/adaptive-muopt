% function [A, g, w] = startEllipse(H, bu, w)
% This function initializes the ellipsoid method for energy minimization
% problem for MAC. Called by minPMAC program
%
% function [A, g] = startEllipse(H, bu, w)
% H containes the channel matrices as is defined in minPMAC.m and bu is
% the target U by 1 rate vector, w is the U by 1 energy weights.
% For all fixed margin WF steps, decoding order is 1,2,...U.

% A is the matrix describing the staring ellipsoid and g is its center

function [A, g, w] = startEllipse(H, bu, w, cb)

    [Ly, U, N] = size(H);

    order = 1:U; % decoding order, arbitrary
    tmax = zeros(U, 1);

    for u = 1:U
        en = zeros(U, N);
        b = bu;
        b(u) = b(u) + 1;

        for u1 = flipud(order) % starting from last user.

            for index = 1:N
                Rnoise(:, :, index) = eye(Ly);
            end

            for u2 = flipud(order) % Note that the users that have been decoded has zero E

                if u2 == u1
                else

                    for index = 1:N
                        Rnoise(:, :, index) = Rnoise(:, :, index) + en(u2, index) * ...
                            H(:, u2, index) * H(:, u2, index)';
                    end

                end

            end

            for index = 1:N
                %h_temp = Rnoise(:,:,index)^(-1/2) * G(:,u1,index);
                h_temp = sqrtm(Rnoise(:, :, index)) \ H(:, u1, index);
                g(u1, index) = svd(h_temp) ^ 2;
            end

            [b_temp, E_temp] = fmwaterfill_gn(g(u1, :), b(u1) / N, 0, cb);
            en(u1, :) = E_temp;
            bn(u1, :) = b_temp;
        end

        tmax(u) = sum(w .* sum(en, 2)); % \sum_n \sum_u \w_u En_u
    end

    g = tmax / 2;
    % if max(g) > 1e6
    %     w = w/max(g)*1e5;
    %     g = g/max(g)*1e5;
    % end
    A = eye(U) * sum(g .^ 2); % an sphere centerced at tmax/2
