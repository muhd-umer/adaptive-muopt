% function [f, b, E] = Lag_dual_f_rate(G, theta, w, Eu , cb);
% this function computes the Lagrange dual function by solving the
% optimization problem (calling the function minPtone) on each tone.
% the inputs are: 
% 1)  H, an Ly by U by N channel matrix. Ly is the number of receiver antennas, 
%     U is the total number of users and N is the total number of tones.
%     H(:,:,n) is the channel matrix for all users on tone n
%     and H(:,u,n) is the channel for user u on tone n. In this code we assume each user 
%     only has single transmit antenna, thus H(:,u,n) is a column vector. 
% 2)  theta, a U by 1 vector containing the weights for the rates.
% 3)  w, a U by 1 vector containing the weights for each user's power.
% 4)  Eu, a U by 1 vector containing the power constraints for all the users
% 5)  cb = 1 for complex bb and cb=2 for real bb
%
% the outputs are:
% 1)  f, the Lagrange dual function value.
% 2)  b, a U by N vector containing the rates for all users and over all tones
%     that optimizes the Lagrangian. b(u,:) is the rate allocation for user
%     u over all tones.
% 3)  E, a U by N vector containing the powers for all users and over all tones
%     that optimizes the Lagrangian. E(u,:) is the power allocation for
%     user u over all tones.
% *************************************************************
function [f, b, E] = Lag_dual_f_rate(H, theta, w, Eu, cb);

[Ly, U, N] = size(H);
f = 0; 
b = zeros(U,N);
E = zeros(U,N);

% Performing optimization over all N tones,
for i = 1:N
    [temp, b(:,i), E(:,i)] = minPtonerate(H(:,:,i), theta, w , cb); 
    f = f + temp;
end
f = f + w' * Eu;