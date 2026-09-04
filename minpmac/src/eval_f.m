% function f = eval_f(theta, G, e, w)
% This function evaluates the value of the function,
%
% f(e) = (theta_1 - theta_2)/2 * log det(I + G_1 * G_1^H * e_1) +
% (theta_2 - theta_3)/2 * log det(I + G_1 * G_1^H * e_1 + G_2 * G_2^H * e_2) + ...
% (theta_{U-1} - theta_U)/2 * log det(I + G_1 * G_1^H * e_1 + ... + G_{U-1} * G_{U-1}^H * e_{U-1}) +
% theta_U/2 * log det(I + G_1 * G_1^H * e_1 + ... + G_U * G_U^H * e_U) - w^T * e + sum_{u=1}^U log(e_u)
% theta should be in decreasing order, theta_1 >= theta_2 >= ... >=theta_U.
%
% the inputs are:
% 1)  theta, a U by 1 vector of weights for the rates.
% 2)  H, an Ly by U channel matrix. G(:,u) is the channel
%     vector for user u. Again each user has just one transmit antenna.
% 3)  e, a U by 1 vector containing each user's power.
% 4)  w, a U by 1 vector containing weights for each user's power

% the output is f the function value given above.
function f = eval_f(theta, H, e, w)
    [Ly, U] = size(H);

    theta = 0.5 * (theta - [theta(2:U); 0]);

    M = zeros(Ly, Ly, U); % M(:,:,i) = (I + sum_{u=1}^i G_u * G_u^H * e_u)
    % M is computed recursively

    M(:, :, 1) = eye(Ly) + H(:, 1) * H(:, 1)' * e(1);
    f = theta(1) * log(det(M(:, :, 1))) + log(e(1)) - w(1) * e(1);

    for u = 2:U
        M(:, :, u) = M(:, :, u - 1) + H(:, u) * H(:, u)' * e(u);
        f = f + theta(u) * log(det(M(:, :, u))) + log(e(u)) - w(u) * e(u);
    end
