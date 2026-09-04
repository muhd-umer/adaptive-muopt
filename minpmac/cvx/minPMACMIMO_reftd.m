function [FEAS_FLAG, bu_a, info] = minPMACMIMO_cvx(H, Lx, bu_min, w, cb)
    % MINPMACMIMO_cvx Minimum power multi-user MIMO MAC with heterogeneous antennas
    %
    % SYNTAX:
    %   [FEAS_FLAG, bu_a, info] = minPMACMIMO_cvx(H, Lx, bu_min, w, cb)
    %
    % INPUTS:
    %   H       - Channel matrix [Ly, sum(Lx), N] concatenated across users
    %   Lx      - Antennas per user [1, U] where sum(Lx) = total TX antennas
    %   bu_min  - Target rates per user [1, U] (bits per channel use)
    %   w       - Energy weights [1, U] (positive weights for energy objective)
    %   cb      - Baseband type: 1=complex, 2=real (affects rate calculation)
    %
    % OUTPUTS:
    %   FEAS_FLAG - Feasibility: 0=infeasible, 1=single order, 2=time-sharing
    %   bu_a      - Achieved rates [1, U] (bits per channel use)
    %   info      - Structure with detailed results
    %
    % EXAMPLE:
    %   Lx = [2, 3, 2]; H = randn(4,7,8) + 1j*randn(4,7,8);  % 4 RX, 7 total TX, 8 tones
    %   [flag, rates, info] = minPMACMIMO_cvx(H, Lx, [3 3 3], [1 1 1], 1);

    tic;
    [Ly, Ltot, N] = size(H);
    U = length(Lx);

    % validate inputs
    if sum(Lx) ~= Ltot
        error('sum(Lx) = %d must equal total TX antennas = %d', sum(Lx), Ltot);
    end

    if length(bu_min) ~= U || length(w) ~= U
        error('bu_min and w must have length U=%d', U);
    end

    if ~all(w > 0)
        error('Energy weights w must be positive');
    end

    if ~all(bu_min >= 0)
        error('Target rates bu_min must be non-negative');
    end

    if ~all(Lx >= 1)
        error('Each user must have at least 1 antenna');
    end

    % compute antenna index mapping
    idx_end = cumsum(Lx);
    idx_start = [1, idx_end(1:end - 1) + 1];

    % convert to column vectors and ensure double precision
    bu_min = double(bu_min(:));
    w = double(w(:));
    H = double(H);

    % scale rates for convex formulation
    bu_scaled = cb * bu_min;

    fprintf('minPMACMIMO: Solving for %d users, %d tones, %d RX antennas\n', U, N, Ly);
    fprintf('Antenna configuration: [%s], total=%d\n', num2str(Lx), Ltot);

    %% solve convex optimization problem
    [b_opt, Rxx_opt, theta, cvx_stat] = solve_cvx_mimo(H, bu_scaled, w, Ly, U, N, Lx, idx_start, idx_end);

    if ~strcmp(cvx_stat, 'Solved') && ~strcmp(cvx_stat, 'Inaccurate/Solved')
        fprintf('CVX failed with status: %s\n', cvx_stat);
        FEAS_FLAG = 0;
        bu_a = zeros(1, U);
        info = struct('sol_status', cvx_stat, 'feasible', false);
        return;
    end

    % rescale rates back
    b_opt = b_opt / cb;
    bu_scaled = bu_scaled / cb;

    %% cluster users by lagrange multipliers
    [clusters, theta_unique] = cluster_theta_values(theta);

    %% generate all possible decoding orders
    all_orders = generate_decoding_orders(clusters);

    fprintf('Found %d clusters, %d possible decoding orders\n', ...
        length(clusters), size(all_orders, 1));

    %% compute solution based on number of orders
    if size(all_orders, 1) == 1
        % single decoding order
        order = all_orders(1, :);
        [bu_achieved, b_achieved] = compute_achieved_rates_mimo(H, Rxx_opt, order, ...
            Ly, U, N, cb, Lx, idx_start, idx_end);

        FEAS_FLAG = 1;
        bu_a = bu_achieved';

        % store single-order results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, {order}, 1, {Rxx_opt}, bu_achieved, b_achieved, cvx_stat, ...
            idx_start, idx_end);

    else
        % multiple orders - solve for time-sharing weights
        [weights, bu_vertices, orderings] = solve_time_sharing_mimo(H, Rxx_opt, ...
            all_orders, bu_min, Ly, U, N, cb, Lx, idx_start, idx_end);

        if isempty(weights)
            FEAS_FLAG = 0;
            bu_a = zeros(1, U);
            info = struct('sol_status', cvx_stat, 'feasible', false, ...
                'time_sharing_failed', true);
            return;
        end

        % compute weighted average of achieved rates
        bu_a = (bu_vertices * weights)';
        FEAS_FLAG = 2;

        % store time-sharing results
        info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
            clusters, orderings, weights, {Rxx_opt}, bu_vertices, [], cvx_stat, ...
            idx_start, idx_end);
    end

    % compute average energies
    info.Eu_avg = compute_average_energies_mimo(Rxx_opt, U, N, Lx, idx_start, idx_end);
    info.elapsed_time = toc;

    fprintf('minPMACMIMO completed in %.3f seconds, FEAS_FLAG=%d\n', info.elapsed_time, FEAS_FLAG);
end

function [b, Rxx, theta, cvx_stat] = solve_cvx_mimo(H, bu_min, w, Ly, U, N, Lx, idx_start, idx_end)
    % solve the convex MIMO minPMAC optimization problem

    Ltot = sum(Lx);

    % generate all non-empty subsets for polymatroid constraints
    subsets = generate_user_subsets(U);

    cvx_begin quiet
    cvx_solver mosek
    variable b(U, N) nonnegative
    variable Rxx(Ltot, Ltot, N) hermitian semidefinite
    dual variable theta

    % rate constraints with dual variables
    theta:sum(b, 2) >= bu_min;

    % MAC polymatroid constraints for all subsets and tones
    for s = 1:length(subsets)
        subset = subsets{s};

        for n = 1:N
            % compute interference covariance matrix for this subset
            interference_cov = zeros(Ly, Ly);

            for u = subset
                ant_idx = idx_start(u):idx_end(u);
                H_u = H(:, ant_idx, n);
                R_u = Rxx(ant_idx, ant_idx, n);
                interference_cov = interference_cov + H_u * R_u * H_u';
            end

            % MAC capacity constraint using log_det (CVX function)
            sum(b(subset, n)) <= log_det(eye(Ly) + interference_cov) / log(2);
        end

    end

    % objective: minimize weighted energy sum
    objective = 0;

    for u = 1:U
        ant_idx = idx_start(u):idx_end(u);

        for n = 1:N
            objective = objective + w(u) * trace(Rxx(ant_idx, ant_idx, n));
        end

    end

    minimize(objective)
    cvx_end

    cvx_stat = cvx_status;

    % ensure Rxx is properly shaped for downstream processing
    if ndims(Rxx) == 2 && N > 1
        Rxx = reshape(Rxx, Ltot, Ltot, N);
    end

    Rxx = full(Rxx); % convert from sparse if needed
end

function subsets = generate_user_subsets(U)
    % generate all non-empty subsets of users {1, 2, ..., U}
    subsets = cell(2 ^ U - 1, 1);
    idx = 1;

    for i = 1:(2 ^ U - 1)
        subset = [];

        for u = 1:U

            if bitget(i, u)
                subset = [subset, u];
            end

        end

        subsets{idx} = subset;
        idx = idx + 1;
    end

end

function [clusters, theta_unique] = cluster_theta_values(theta)
    % cluster users by their lagrange multiplier values
    tol = 1e-8;
    theta_vec = double(theta(:));

    % find unique theta values within tolerance
    [theta_unique, ~, groups] = uniquetol(theta_vec, tol);

    % group users by theta values
    num_clusters = length(theta_unique);
    clusters = cell(num_clusters, 1);

    for k = 1:num_clusters
        clusters{k} = find(groups == k)';
    end

end

function orders = generate_decoding_orders(clusters)
    % generate all possible decoding orders considering clusters
    if length(clusters) == 1
        % single cluster - all permutations
        orders = perms(clusters{1});
    else
        % multiple clusters - cartesian product of permutations
        cluster_perms = cell(length(clusters), 1);

        for k = 1:length(clusters)
            cluster_perms{k} = perms(clusters{k});
        end

        % compute cartesian product
        orders = cluster_perms{1};

        for k = 2:length(clusters)
            new_orders = [];

            for i = 1:size(orders, 1)

                for j = 1:size(cluster_perms{k}, 1)
                    new_orders = [new_orders; [orders(i, :), cluster_perms{k}(j, :)]];
                end

            end

            orders = new_orders;
        end

    end

end

function [bu_achieved, b_achieved] = compute_achieved_rates_mimo(H, Rxx, order, ...
        Ly, U, N, cb, Lx, idx_start, idx_end)
    % compute achieved rates for given decoding order using MIMO SIC

    b_achieved = zeros(U, N);

    for n = 1:N
        Rn = Rxx(:, :, n);

        for k = 1:U
            u = order(k);

            % signal + interference covariance (users decoded from position k to U)
            signal_plus_intf = zeros(Ly, Ly);

            for j = k:U
                v = order(j);
                ant_idx = idx_start(v):idx_end(v);
                H_v = H(:, ant_idx, n);
                R_v = Rn(ant_idx, ant_idx);
                signal_plus_intf = signal_plus_intf + H_v * R_v * H_v';
            end

            % interference only (users decoded after position k)
            interference = zeros(Ly, Ly);

            for j = (k + 1):U
                v = order(j);
                ant_idx = idx_start(v):idx_end(v);
                H_v = H(:, ant_idx, n);
                R_v = Rn(ant_idx, ant_idx);
                interference = interference + H_v * R_v * H_v';
            end

            % compute rate using difference of log determinants
            rate_with_signal = log2(det(eye(Ly) + signal_plus_intf));
            rate_interference_only = log2(det(eye(Ly) + interference));
            
            b_achieved(u, n) = real(rate_with_signal - rate_interference_only) / cb;
        end

    end

    bu_achieved = sum(b_achieved, 2);
end

function [weights, bu_vertices, orderings] = solve_time_sharing_mimo(H, Rxx, orders, ...
        bu_min, Ly, U, N, cb, Lx, idx_start, idx_end)
    % solve for optimal time-sharing weights between decoding orders

    num_orders = size(orders, 1);
    bu_vertices = zeros(U, num_orders);
    orderings = cell(num_orders, 1);

    % compute achieved rates for each order
    for k = 1:num_orders
        orderings{k} = orders(k, :);
        [bu_achieved, ~] = compute_achieved_rates_mimo(H, Rxx, orders(k, :), ...
            Ly, U, N, cb, Lx, idx_start, idx_end);
        bu_vertices(:, k) = bu_achieved;
    end

    % solve for time-sharing weights
    tol = 1e-3;
    cvx_begin quiet
    cvx_solver mosek
    variable weights(num_orders) nonnegative
    variable z(num_orders) binary

    % minimize number of active vertices
    minimize(sum(z))

    subject to
    % weights sum to 1
    sum(weights) == 1;

    % achieve target rates
    bu_vertices * weights >= bu_min - tol;

    % sparsity constraint
    weights <= z;
    cvx_end

    if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
        weights = [];
        return;
    end

    % keep only non-zero weights
    active_idx = weights > 1e-8;
    weights = weights(active_idx);
    weights = weights / sum(weights); % renormalize
    bu_vertices = bu_vertices(:, active_idx);
    orderings = orderings(active_idx);
end

function Eu_avg = compute_average_energies_mimo(Rxx, U, N, Lx, idx_start, idx_end)
    % compute average energy per user across tones
    Eu_avg = zeros(1, U);

    for u = 1:U
        ant_idx = idx_start(u):idx_end(u);
        total_energy = 0;

        for n = 1:N
            total_energy = total_energy + trace(Rxx(ant_idx, ant_idx, n));
        end

        Eu_avg(u) = total_energy / N;
    end

end

function info = create_info_struct_mimo(H, Lx, bu_min, w, cb, theta, theta_unique, ...
        clusters, orderings, weights, Rxx_cell, bu_vertices, b_achieved, cvx_status, ...
        idx_start, idx_end)
    % create comprehensive info structure for MIMO case

    info = struct();
    info.H = H;
    info.Lx = Lx;
    info.bu_min = bu_min;
    info.w = w;
    info.cb = cb;
    info.theta = theta;
    info.theta_unique = theta_unique;
    info.clusters = clusters;
    info.orderings = orderings;
    info.weights = weights;
    info.Rxx = Rxx_cell;
    info.bu_vertices = bu_vertices;
    info.b_achieved = b_achieved;
    info.sol_status = cvx_status;
    info.feasible = true;
    info.idx_start = idx_start;
    info.idx_end = idx_end;
end
