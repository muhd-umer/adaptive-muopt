% function [E, w, b] = maxRMAC_src(H, Eu, theta, cb)
%
% maxRMAC_src Maximizes weighted rate sum subject to energy constraint, each
% user has ONLY ONE transmit antenna (Lxu=1). It does not use CVX.
%
%   INPUTS:
% H(:,u,n): Ly x U x N channel matrix. Ly = number of receiver antennas,
%     U = number of users and N = number of tones.
%
%     If the channel is real-bbd (cb=2), maxPMAC_cvx realizes user data rates
%     over all the tones (or equivalently positive and negative
%     freqs). This means the input H must be conjugate symmetric H_n =
%     H_{N-n}^*.  The program will reduce N by 2 and focus energy on the
%     lower half of frequencies. Thus N>1 must be even for real channels.
%     For real N=1 channel, the program runs and will produce half the data
%     rates for the same complex N=1 channel.
%
%     By constast if cb=1 (cplx bbd), maxRMAC_cvx need not have a conjugate
%     symmetric H and N is not reduced.
%
% Eu: 1 x U Energy constraint for each user as total energy per symbol.
%
% theta: weight on each user's rate, length-U vector.
%
% cb: =1 if H is complex baseband, and cb=2 if H is real baseband.
%
%
%   OUTPUTS:
% E:    U x N energy distribution. Eun(u,n) is user u's energy allocation
%         on tone n.
% w:   U x 1 Lagrangian multiplier w.r.t. energy constraints
% b: U by N bit distributions for all users.
%
% Subroutines called
%     startEllipseRate
%     Lag_dual_f_rate
%     minPtonerate
% **********************************************************************
function [E, w, b] = maxRMAC_src(H, Eu, theta, cb)

    err = 1e-9; % error tolerance
    count = 0;
    [Ly, U, N] = size(H);

    if N > 1

        if cb == 2
            H = H(:, :, 1:N / 2);
            N = N / 2;
        end

    end

    [A, g] = startEllipseRate(H, Eu, theta); % starting ellipsoid
    w = g;

    while 1
        % Ellipsoid method starts here
        [f, b, E] = Lag_dual_f_rate(H, theta, w, Eu, cb);
        g = Eu - sum(E, 2); % sub-gradient

        if sqrt(g' * A * g) <= err % stopping criteria
            break
        end

        % Updating the ellipsoid
        gt = g / sqrt(g' * A * g);
        w = w - 1 / (U + 1) * A * gt;
        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * A * gt * gt' * A);

        ind = find(w < zeros(U, 1));

        while ~isempty(ind) % This part is to make sure that w is feasible,
            g = zeros(U, 1); % it was not covered in the lecture notes and you may skip this part
            g(ind(1)) = -1;
            gt = g / sqrt(g' * A * g);
            w = w - 1 / (U + 1) * A * gt;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * A * gt * gt' * A);
            ind = find(w < zeros(U, 1));
        end

        count = count + 1;
    end

    b = b / log(2); % conversion from nuts to bits
