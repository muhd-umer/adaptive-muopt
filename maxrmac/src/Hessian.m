% function [g, H] = Hessian(theta, G, e, w)
%
% This function calculates the gradiant (g) and the Hessian (H) of the
% function f(e) = (theta_1 - theta_2)/2 * log det(I + G_1 * G_1^H * e_1) + 
% (theta_2 - theta_3)/2 * log det(I + G_1 * G_1^H * e_1 + G_2 * G_2^H * e_2) + ...
% (theta_{U-1} - theta_U)/2 * log det(I + G_1 * G_1^H * e_1 + ... + G_{U-1} * G_{U-1}^H * e_{U-1}) +
% theta_U/2 * log det(I + G_1 * G_1^H * e_1 + ... + G_U * G_U^H * e_U) - w^T * e + sum_{u=1}^U log(e_u)
% theta should be in decreasing order, theta_1 >= theta_2 >= ... >=theta_U.
%
% the inputs are:
% 1)  theta, a U by 1 vector of weights for the rates.
% 2)  G, an Ly by U channel matrix. G(:,u) is the channel 
%     vector for user u. Again each user has just one transmit antenna.
% 3)  e, a U by 1 vector containing each user's power.
% 4)  w, a U by 1 vector containing weights for each user's power
%
% the outputs are:
% 1)  g, U by 1 gradiant vector.
% 2)  H, U by U Hessian matrix.
function [g, H] = Hessian(theta, G, e, w)
[Ly,U] = size(G);

theta = 0.5 * (theta - [theta(2:U); 0]);

M = zeros(Ly,Ly,U);                         % M(:,:,i) = (I + sum_{u=1}^i G_u * G_u^H * e_u)^{-1}
                                          % M is computed recursively using matrix inversion lemma

M(:,:,1) = eye(Ly) - G(:,1) * G(:,1)' * e(1) / (1 + e(1) * G(:,1)' * G(:,1));

for u = 2:U
    M(:,:,u) = M(:,:,u-1) - M(:,:,u-1) * G(:,u) * G(:,u)' * M(:,:,u-1) * e(u) / (1 + e(u) * G(:,u)' * M(:,:,u-1) * G(:,u));
end

g = zeros(U,1);

% g_u = sum_{j=u}^U theta_j * G_u * M_j * G_u^H - w_u + 1/e_u
for u = 1:U
    for j = u:U
        g(u) = g(u) + theta(j) * G(:,u)' * M(:,:,j) * G(:,u);
    end
end

g = g + 1./ e - w;

% H_{u,l} = sum_{j = max(u,l)}^U -theta_j * tr(G_u * G_u^H * M_j * G_l * G_l^H * M_j) - 1/e_u^2 * delta(u-l) 

H = zeros(U,U);
for u = 1:U
    for l = 1:U
        for j = max(u,l):U
            H(u,l) = H(u,l) - theta(j) * trace(G(:,u) * G(:,u)' * M(:,:,j) * G(:,l) * G(:,l)' * M(:,:,j));
        end
    end
end
H = H - diag(1./(e.^2));


