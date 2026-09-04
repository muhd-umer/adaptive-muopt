% function [Rxxs, Eun, w, bun] = maxRMACMIMO_cvx(H, Lxu, Eu, theta , cb)
%
% maxRMAC_vector_cvx Maximize weighted rate sum subject to energy
% constaint. This uses cvx and mosek.
%
%   INPUTS:
%     H(:,u,n): Ly x U x N channel matrix. Ly = number of receiver antennas,
%     U = number of users and N = number of tones.
%
%     If the channel is real-bbd (cb=2), maxPMAC_cvx realizes user data rates
%     over all the tones (or equivalently positive and negative
%     freqs). This means the input H must be conjugate symmetric H_n =
%     H_{N-n}^*.  The program will reduce N by 2 and focus energy on the
%     lower half of frequencies. Thus N>1 must be even for real channels,
%     with a special exception made for N=1.
%
%     By constast if cb=1 (cplx bbd), maxRMAC_cvx need not have a conjugate
%     symmetric H and N is not reduced.
%
% Eu: 1 x U Energy constraint for each user as total energy per symbol.
%     For N=1 channel, the program adjusts energy to be per complex (cb=1)
%     symbol at the beginning and then restores at end.  cb=2 has no
%     change.  So for N=1, the input Eu should be u's energy/symbol.
%
% theta: weight on each user's rate, length-U vector.
%
% cb: =1 if H is complex baseband, and cb=2 if H is real baseband.
%
%   OUTPUTS:
% Rxxs:  U-by-N cell array containing Rxx(u,n)'s if Lxu is a length-U
%        vector; or Lxu-by-Lxu-by-U-by-N tensor if Lxu is a scalar.
% Eun:    U x N energy distribution. Eun(u,n) is user u's energy allocation
%         on tone n.
% w:      U x 1 Lagrangian multiplier w.r.t. energy constraints
% bun:    U by N bit distributions for all users.
% bun: U by N bit distributions for all users.
% **********************************************************************
function [Rxxs, Eun, w, bun] = maxRMACMIMO_cvx(H, Lxu, Eu, theta, cb)

    UNIFORM_FLAG = 0;
    [Ly, ~, N] = size(H);

    if N > 1

        if cb == 2
            H = H(:, :, 1:N / 2);
            N = N / 2;
        end

    else
        Eu = Eu / (3 - cb);
        H = reshape(H, Ly, [], 1);
    end

    Eu = reshape(Eu, [], 1);
    theta = reshape(theta, [], 1);
    [stheta, idx] = sort(theta, 'descend');
    Eu = Eu(idx);
    delta = -diff([stheta; 0]);
    U = length(Eu);

    if length(Lxu) == 1
        Lxu = ones(1, U) * Lxu;
        UNIFORM_FLAG = 1;
    end

    Lxu_max = max(Lxu);
    index_end = cumsum(Lxu);
    index_start = [1, index_end(1:end - 1) + 1];
    Lxu = Lxu(idx);
    index_start = index_start(idx);
    index_end = index_end(idx);
    cvx_begin quiet
    cvx_solver mosek
    variable rxxs(Lxu_max, Lxu_max, U, N) hermitian
    dual variable w
    expressions r(U, N) S(Ly, Ly) Eun(U, N)

    for n = 1:N
        S = eye(Ly);

        for u = 1:U
            S = S + H(:, index_start(u):index_end(u), n) ...
                * rxxs(1:Lxu(u), 1:Lxu(u), u, n) ...
                * H(:, index_start(u):index_end(u), n)';
            r(u, n) = log_det(S);
            Eun(u, n) = trace(rxxs(1:Lxu(u), 1:Lxu(u), u, n));
        end

    end

    maximize sum(delta' * r)
    subject to
    w:sum(Eun, 2) <= Eu;

    for u = 1:U

        for n = 1:N
            rxxs(1:Lxu(u), 1:Lxu(u), u, n) == hermitian_semidefinite(Lxu(u));
        end

    end

    cvx_end

    cumrate = zeros(U, N);

    for n = 1:N
        S = eye(Ly);

        for u = 1:U
            S = S + H(:, index_start(u):index_end(u), n) ...
                * rxxs(1:Lxu(u), 1:Lxu(u), u, n) ...
                * H(:, index_start(u):index_end(u), n)';
            cumrate(u, n) = (1 / cb) * real(log2(det(S)));
        end

    end

    bun(idx, :) = diff([zeros(1, N); cumrate]);
    w(idx) = w;

    if N == 1
        Eun = (3 - cb) * Eun;
        rxxs = (3 - cb) * rxxs;
    end

    if UNIFORM_FLAG
        Rxxs(:, :, idx, 1:N) = rxxs;

        for u = 1:U

            for n = 1:N
                Eun(idx(u), n) = trace(Rxxs(:, :, idx(u), n));
            end

        end

    else
        Rxxs = cell(U, N);

        for u = 1:U

            for n = 1:N
                Rxxs{idx(u), n} = rxxs(1:Lxu(u), 1:Lxu(u), u, n);
                Eun(idx(u), n) = trace(Rxxs{idx(u), n});
            end

        end

    end

end
