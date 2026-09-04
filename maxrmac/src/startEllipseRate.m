% function [A, g] = startEllipseRate(H, Eu, theta)
% This function initializes the ellipsoid method for rate maximization problem.  
%
% function [A, g] = startEllipseRate(G, Eu, theta)
% H containes the channel matrices as is defined in maxRMAC.m and Eu is 
% the U by 1 power contraint vector, theta is the U by 1 rate weights. 
% For all rate maximization WF steps, decoding order is 1,2,...U.
%
% A is the matrix describing the staring ellipsoid and g is its center
% *********************************************************************
function [A, g] = startEllipseRate(H, Eu, theta)

[Ly, U, N] = size(H);

order = [1:U];                       % decoding order, arbitrary
tmax = zeros(U,1);

for u = 1:U
    bn = zeros(U,N);
    en = zeros(U,N);
    E = Eu;
    E(u) = E(u) * 0.9;
    for u1 = flipud(order)           % starting from last user.
      
        for index = 1:N
            Rnoise(:,:,index) = eye(Ly);
        end
        for u2 = flipud(order)       % Note that the users that have been decoded has zero E
            if u2 == u1,
            else 
                for index = 1:N
                    Rnoise(:,:,index) = Rnoise(:,:,index) + en(u2,index) * ...
                                        H(:,u2,index) * H(:,u2,index)';
                end
            end
        end
        for index = 1:N
            h_temp = Rnoise(:,:,index)^(-1/2) * H(:,u1,index);
            g(u1,index) = svd(h_temp)^2;
        end
        
        [b_temp, E_temp] = waterfill_gn(g(u1,:), Eu(u1) / N, 0);
        
        en(u1,:) = E_temp;
        bn(u1,:) = b_temp;
    end
    tmax(u) = sum(theta .* sum(bn,2));    % \sum_n \sum_u \theta_u bn_u
end
A = eye(U) * sum(tmax.^2) / 4;        % an sphere centerced at tmax/2
g = tmax / 2;





    
    