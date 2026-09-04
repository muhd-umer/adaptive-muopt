% function [Eun, theta, bun, FEAS_FLAG, bu_a, info] = minPMAC(H, bu, w, cb)
%
% This main function contains ellipsoid method part and calls direclty or
%   indirectly 6 other functions.  minPMAC uses no CVX.  Another routine
%   minPMAC_cvx uses cvx and usually runs longer.  However, these scalar
%   minPMAC programs assume each user has Lxu = 1;
%   There is a CVX-using program minPMACMIMO that allows variable Lxu.
%
% INPUTS:
% H(:,u,n): Ly x U x N channel matrix. Ly = number of receiver antennas,
%     U = number of users and N = number of tones.
%
%     If the channel is real-bbd (cb=2), minPMAC realizes user data rates
%     over the lower half of the tones (or equivalently, the positive
%     freqs), and directly corresponds to the input bu user-data rate vector.
%     N is the number of tones actually transmitted (so corresponds to a 2N
%     size FFT when cb=2).
%
%     By constast if cb=1 (cplx bbd), minPMAC realizes user data rates
%     over all tones, but uses the same core optmization, so the data rate
%     is halved internal to the program, realizing that half bu rate on the
%     lower tones, because the designer (cb=1) wants it over all tones.
%     N is the number of tones actually transmitted (so corresponds to a N
%     size full-complex FFT when cb=1).
%
% bu_min: U x 1 vector containing the target rates for all the users.
%
% w:  U x 1 vector containing the weights for each user's power.
%
% cb: =1 if H is complex baseband, and cb=2 if H is real baseband.
%
% OUTPUTS:
% Eun:    U by N energy distribution that minimizes the weighted-sum energy.
%       E(u,n) is user u's energy allocation on tone n.
% theta:the optimal U by 1 dual variable vector containing optimal weights
%     of rates. Theta determines the decoding order. Largest theta is
%     decoded last, and smallest first.
% bun,   U by N bit distributions for all users.
% FEAS_FLAG: indicator of achievability.
%           FEAS_FLAG=1 if the target is achieved by a single ordering;
%           FEAS_FLAG=2 if the target is achieved by time-sharing
% bu_a: U-by-1 vector showing achieved sum rate of each user.
% info: various length output depending on FEAS_FLAG
%           --if FEAS_FLAG=1: 1 x 4 cell array containing
%               {Rxxs, Eun, bun, theta} corresponds to the single vertex
%               there are no equal-theta user sets in this case
%           --if FEAS_FLAG=2: 1-x 6 cell array, with each row representing
%               a time-shared vertex {Rxxs, Eun, bun, theta, frac}
%               there are numclus equal-theta user sets in this case.
%
%  info's row entries in detail (one row for each vertex shared
%       - Eun:  U-by-N matrix showing users' transmit energy on each tone.
%           If infeasible, output 0.
%       - bun: U-by-N matrix showing users' rate on each tone. If
%           infeasible, output 0.
%       - theta: U-by-1 Lagrangian multiplier w.r.t. target rates
%       - order: produces the order from left(best) to right for vertex
%       - frac: fraction of dimensions for each vertex in time share (FF =2
%           ONLY)
%       - cluster: index (to which cluster the user belongs; 0 means no
%           cluster)
%
% Subroutines called directly are
%       startEllipse.m
%       Lag_dual_f.m
%   and indirectly
%       minPtone.m
%       Hessian.m
%       eval_f.m
%       fmwaterfill_gn.,
%
% ***************************************************************************
function [Eun, theta, bun, FEAS_FLAG, bu_a, info] = minPMAC(H, bu_min, w, cb)

    tstart = tic;
    bu_min = 1 / (3 - cb) * bu_min;
    err = 1e-9;
    conv_tol = 1e-2; % error tolerance

    count = 0;
    [Ly, U, N] = size(H);
    w = reshape(w, U, 1);
    bu_min = reshape(bu_min, U, 1);
    bun = zeros(U, N);
    Eun = zeros(U, N);

    [A, g, w] = startEllipse(H, bu_min, w, cb); % starting ellipsoid
    theta = g;

    bu_min = bu_min * log(2); % conversion from bits to nuts

    while 1
        %     disp(count)
        % Ellipsoid method starts here
        [~, bun, Eun] = Lag_dual_f(H, theta, w, bu_min);
        g = sum(bun, 2) - bu_min; % sub-gradient

        if sqrt(g' * A * g) <= err % stopping criteria
            break
        end

        % Updating the ellipsoid
        tmp = A * g / sqrt(g' * A * g);
        theta = theta - 1 / (U + 1) * tmp;

        A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));

        ind = find(theta < zeros(U, 1));

        while ~isempty(ind) % This part is to make sure that theta is feasible,
            g = zeros(U, 1); % it was not covered in the lecture notes and you may skip this part
            g(ind(1)) = -1;
            tmp = A * g / sqrt(g' * A * g);
            theta = theta - 1 / (U + 1) * tmp;
            A = U ^ 2 / (U ^ 2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
            ind = find(theta < zeros(U, 1));
        end

        count = count + 1;
    end

    bun = (3 - cb) * bun / log(2); % conversion from nats to bits

    %-------------------------------
    bu_min = (3 - cb) * bu_min / log(2);
    bu_a = sum(bun, 2);

    % --------- REORDER USERS ACCORDING TO THETA -------
    % This ordering will need eventually to be reversed before returing from
    % this function.
    [theta, Itheta] = sort(theta, 'descend');
    [~, Jtheta] = sort(Itheta);
    % theta = theta(Jtheta) reverses this sort later.
    bu_a = bu_a(Itheta);
    bun = bun(Itheta, :);
    Eun = Eun(Itheta, :);
    bu_min = bu_min(Itheta);
    H = H(:, Itheta, :);

    % --------------  SIMPLE CASE OF NO EQUAL-THETA USERS -------------
    % This next section is implemented only if there are no equal-theta users
    % and then returns results.
    if isempty(find(abs(diff(theta)) <= 1e-5 * min(theta), 1))
        FEAS_FLAG = 1;
        bu_a = sum(bun, 2);
        % restore the original order
        bu_a = bu_a(Jtheta);
        bu_v = bu_a';
        theta = theta(Jtheta);
        bun = bun(Jtheta, :);
        Eun = Eun(Jtheta, :);
        % make table and exit
        info = table(bu_v, {Eun}, {bun}, {theta}, {Itheta(end:-1:1)}); % detailed info of boundary vertices
        info.Properties.VariableNames(2:end) = {'Eun' 'bun' 'theta', 'order'};
        toc(tstart)
        return % ARRIVE HERE AND SIMPLE CASE IS DONE.PROGRAM OVER
    end

    %
    %---------------------TWO OR MORE EQUAL-THETA USERS ------------------
    % This point of program is only reached if there are equal theta values for
    % two or more users.  A convex combination of those orders different rate
    % vectors will implement vertex sharing.  The info table above does not
    % exist if we are in this section.  So it will be initialized and then
    % vertices added to it.

    % Only need vertices corresponding to the number of equal-theta users,
    % which number will be the sizeset+1 that this section generates.  Thus,
    % there is no need for U! orders to be searched (which is usually a much
    % larger number.

    % ----------- ENUMERATION OF POSSIBLE ORDERS ---------------
    % there will be setsize+1 orders to check when this section completes as
    % the top setsize+1 rows of the matrix order.
    % invorder restores the original order

    spots = diff([-theta', 0]) < 1e-7 * norm(theta);
    sizeclus = []; % size of each cluster
    numclus = 0; % number of clusters (>1 equal-theta groups)
    set_thetaeq = []; % index of the first entry of each cluster

    i = 1;

    while i <= U
        sizeset = 1;
        flagclus = 0;

        while spots(i) == 1
            sizeset = sizeset + 1;

            if ~flagclus % enter a new cluster, save the last user
                set_thetaeq = [set_thetaeq, i];
                flagclus = 1;
                numclus = numclus + 1;
            end

            i = i + 1;
        end

        if sizeset > 1
            sizeclus = [sizeclus, sizeset];
        end

        i = i + 1;
    end

    order = repmat(1:U, sum(sizeclus - 1), 1);
    cumsize = cumsum([0, sizeclus - 1]);

    for jdx = 1:numclus
        Jright = eye(sizeclus(jdx));
        Jright = [Jright(:, end), Jright(:, 1:end - 1)];
        Jleft = Jright';
        u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1; % users in cluster jdx

        for i = 1:sizeclus(jdx) - 1
            order(cumsize(jdx) + i, u_range) = u_range * Jleft ^ (i); % all permutated orders (excluding 1,2,...U in each cluster)
        end

    end

    % ----------- FIND FIRST VERTEX & CREATE INFO TABLE ----------
    bu_v = sum(bun, 2)'; % 1xU
    initialbu_v = bu_v;
    initialbun = bun;
    firstvertices = de2bi(0:2 ^ U - 1) .* initialbu_v;

    % The vertices will be stored in a table that is indexed by bu_v
    initialinfo = table(bu_v, {Eun}, {bun}, {theta}, {U:-1:1}); % detailed info of boundary vertices
    initialinfo.Properties.VariableNames(2:end) = {'Eun' 'bun' 'theta', 'order'};

    % ----- For each cluster -----------
    for jdx = 1:numclus

        if jdx == 1
            Sinit = repmat(eye(Ly), 1, 1, N);
            cumrateinit = zeros(1, N);

            for n = 1:N

                for i = 1:set_thetaeq(1) - 1
                    Sinit(:, :, n) = Sinit(:, :, n) + H(:, i, n) * Eun(i, n) * H(:, i, n)';
                end

                cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
            end

        else

            for n = 1:N

                for i = set_thetaeq(jdx - 1):set_thetaeq(jdx - 1) + sizeclus(jdx) - 1
                    Sinit(:, :, n) = Sinit(:, :, n) + H(:, i, n) * Eun(i, n) * H(:, i, n)';
                end

                cumrateinit(n) = (1 / cb) * real(log2(det(Sinit(:, :, n))));
            end

        end

        u_range = set_thetaeq(jdx):set_thetaeq(jdx) + sizeclus(jdx) - 1;
        bu_v = initialbu_v;
        known_vertices = initialinfo; % detailed info of boundary vertices
        bd_vertices = bu_v(u_range); % track critical boundary vertices
        vertices = de2bi(0:2 ^ sizeclus(jdx) - 1) .* bd_vertices;

        if min(bd_vertices - bu_min(u_range)') >= -conv_tol % achievable w/o time share
            info = known_vertices;
            info.frac = 1;
            info.clusterID = jdx;

            if jdx == 1
                Big_info = info;
            else
                Big_info = [Big_info; info];
            end

            continue;
        end

        bd_V = 1;

        for idx = cumsize(jdx) + 1:cumsize(jdx) + sizeclus(jdx) - 1 % add more vertices in cluster jdx
            cumrate = zeros(sizeclus(jdx) + 1, N);
            cumrate(1, :) = cumrateinit;

            for n = 1:N
                rel_idx = 2;
                S = Sinit(:, :, n);

                for u = u_range
                    u_or = order(jdx, u);
                    S = S + H(:, u_or, n) * Eun(u_or, n) * H(:, u_or, n)';
                    cumrate(rel_idx, n) = (1 / cb) * real(log2(det(S)));
                    rel_idx = rel_idx + 1;
                end

            end

            bun = initialbun;
            bun(u_range, :) = diff(cumrate);
            bun(order(idx, :), :) = bun;
            bu_v = sum(bun, 2)';

            bd_V = bd_V + 1;
            bd_vertices_extend = [bd_vertices; bu_v(u_range)];
            known_vertices = [known_vertices; {bu_v, {Eun}, {bun}, {theta}, {order(idx, end:-1:1)}}];
            vertices = [vertices; de2bi(0:2 ^ sizeclus(jdx) - 1) .* bu_v(u_range)];
            tess = convhulln(vertices);
            vertices = vertices(unique(tess), :);
            bd_vertices = intersect(bd_vertices_extend, vertices, 'rows', 'stable');
            tess = convhulln(vertices);
            % delete inner vertices
            if size(bd_vertices, 1) < bd_V
                to_remove = setdiff(bd_vertices_extend, bd_vertices, 'rows');
                known_vertices(ismember(known_vertices.bu_v(u_range), to_remove, 'rows'), :) = [];
                bd_V = size(bd_vertices, 1);
            end

            if inhull(bu_min(u_range)', vertices, tess, conv_tol) % bu_min achievable by time-share
                FEAS_FLAG = 2;
                frac = bd_vertices' \ bu_min(u_range);
                a_v = find(frac >= err); % active vertices in time-share
                info = known_vertices(a_v, :);
                info.frac = frac(a_v);
                info.clusterID = ones(length(a_v), 1) * jdx;
                bu_a(u_range) = bd_vertices(a_v, :)' * frac(a_v);
                break;
            end

        end

        % write big info table
        if jdx == 1
            Big_info = info;
        else
            Big_info = [Big_info; info];
        end

    end

    try
        info = Big_info;
    catch ME
        info = info;
    end

    bd_V = size(info.bu_v, 1);

    % ----------- RESTORE ORIGINAL ORDER TO ALL ----------------
    theta = theta(Jtheta);
    temptheta = reshape(cell2mat(info.theta), U, bd_V);
    temptheta = temptheta(Jtheta, :);
    info.theta = mat2cell(temptheta', [ones(1, bd_V)], U);
    %
    tempbun = reshape(cell2mat(info.bun), U, bd_V, N);
    tempbun = tempbun(Jtheta, :, :);
    tempbun = permute(tempbun, [2 1 3]);
    info.bun = mat2cell(tempbun, [ones(1, bd_V)], U, N);
    %
    tempEun = reshape(cell2mat(info.Eun), U, bd_V, N);
    tempEun = tempEun(Jtheta, :, :);
    tempEun = permute(tempEun, [2 1 3]);
    info.Eun = mat2cell(tempEun, [ones(1, bd_V)], U, N);
    %
    bu_a = bu_a(Jtheta);
    info.bu_v = info.bu_v(:, Jtheta);
    %
    tmporder = cell2mat(info.order);
    tmporder = Itheta(tmporder);
    info.order = mat2cell(tmporder, ones(1, bd_V), U);

    if N == 1
        bun = zeros(1, U);

        for i = 1:numel(info.frac)
            bun = bun + squeeze(info.bun{i}) * info.frac(i);
        end

    else
        bun = zeros(U, N);

        for i = 1:numel(info.frac)
            bun = bun + squeeze(info.bun{i}) * info.frac(i);
        end

    end

    toc(tstart)
end
