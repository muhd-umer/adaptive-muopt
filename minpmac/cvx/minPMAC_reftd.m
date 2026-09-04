function [FEAS_FLAG, bu_a, info] = minPMAC_reftd(H, bu_min, w, cb)
    % MINPMAC_REFTD Minimum power multi-user MAC for SISO channels
    %
    % SYNTAX:
    %   [FEAS_FLAG, bu_a, info] = minPMAC_reftd(H, bu_min, w, cb)
    %
    % INPUTS:
    %   H       - Channel matrix [Ly, U, N] where Ly=RX antennas, U=users, N=tones
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
    %   H = randn(2,3,4) + 1j*randn(2,3,4);  % 2 RX, 3 users, 4 tones
    %   [flag, rates, info] = minPMAC_reftd(H, [2 2 2], [1 1 1], 1);

    tic;
    [Ly, U, N] = size(H);

    % validate inputs
    if length(bu_min) ~= U || length(w) ~= U
        error('bu_min and w must have length U=%d', U);
    end

    if ~all(w > 0)
        error('Energy weights w must be positive');
    end

    if ~all(bu_min >= 0)
        error('Target rates bu_min must be non-negative');
    end

    % convert to column vectors and ensure double precision
    bu_min = double(bu_min(:));
    w = double(w(:));
    H = double(H);

    % scale rates for convex formulation (see lecture notes)
    bu_scaled = cb * bu_min;

    fprintf('minPMAC: Solving for %d users, %d tones, %d RX antennas\n', U, N, Ly);

    %% solve convex optimization problem
    [b_opt, Rxx_opt, theta, cvx_stat] = solve_cvx_minpmac(H, bu_scaled, w, Ly, U, N);

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
        [bu_achieved, b_achieved] = compute_achieved_rates(H, Rxx_opt, order, Ly, U, N, cb);

        FEAS_FLAG = 1;
        bu_a = bu_achieved';

        % store single-order results
        info = create_info_struct(H, bu_min, w, cb, theta, theta_unique, clusters, ...
            {order}, 1, {Rxx_opt}, bu_achieved, b_achieved, cvx_stat);

    else
        % multiple orders - solve for time-sharing weights
        [weights, bu_vertices, orderings] = solve_time_sharing(H, Rxx_opt, all_orders, ...
            bu_min, Ly, U, N, cb);

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
        info = create_info_struct(H, bu_min, w, cb, theta, theta_unique, clusters, ...
            orderings, weights, {Rxx_opt}, bu_vertices, [], cvx_stat);
    end

    % compute average energies
    info.Eu_avg = compute_average_energies(Rxx_opt, U, N);
    info.elapsed_time = toc;

    fprintf('minPMAC completed in %.3f seconds, FEAS_FLAG=%d\n', info.elapsed_time, FEAS_FLAG);
end

function [b, Rxx, theta, cvx_stat] = solve_cvx_minpmac(H, bu_min, w, Ly, U, N)
    % solve the convex minPMAC optimization problem

    % generate all non-empty subsets for polymatroid constraints
    subsets = generate_user_subsets(U);

    cvx_begin quiet
    cvx_solver mosek
    variable b(U, N) nonnegative
    variable Rxx(U, N) nonnegative % SISO: scalar power per user per tone
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
                h_u = H(:, u, n); % channel vector for user u, tone n
                interference_cov = interference_cov + Rxx(u, n) * (h_u * h_u');
            end

            % MAC capacity constraint using log_det (CVX function)
            sum(b(subset, n)) <= log_det(eye(Ly) + interference_cov) / log(2);
        end

    end

    % objective: minimize weighted energy sum
    objective = 0;

    for u = 1:U
        objective = objective + w(u) * sum(Rxx(u, :));
    end

    minimize(objective)
    cvx_end

    cvx_stat = cvx_status;
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
    % users within same cluster can be in any order

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

function [bu_achieved, b_achieved] = compute_achieved_rates(H, Rxx, order, Ly, U, N, cb)
    % compute achieved rates for given decoding order using SIC

    b_achieved = zeros(U, N);

    for n = 1:N

        for k = 1:U
            u = order(k);

            % signal covariance for user u
            h_u = H(:, u, n);
            signal_cov = Rxx(u, n) * (h_u * h_u');

            % interference from users decoded after u
            interference_cov = zeros(Ly, Ly);

            for j = (k + 1):U
                v = order(j);
                h_v = H(:, v, n);
                interference_cov = interference_cov + Rxx(v, n) * (h_v * h_v');
            end

            % compute rate using log determinants
            rate_with_signal = log2(det(eye(Ly) + signal_cov + interference_cov));
            rate_interference_only = log2(det(eye(Ly) + interference_cov));
            
            b_achieved(u, n) = real(rate_with_signal - rate_interference_only) / cb;
        end

    end

    bu_achieved = sum(b_achieved, 2);
end

function [weights, bu_vertices, orderings] = solve_time_sharing(H, Rxx, orders, bu_min, Ly, U, N, cb)
    % solve for optimal time-sharing weights between decoding orders

    num_orders = size(orders, 1);
    bu_vertices = zeros(U, num_orders);
    orderings = cell(num_orders, 1);

    % compute achieved rates for each order
    for k = 1:num_orders
        orderings{k} = orders(k, :);
        [bu_achieved, ~] = compute_achieved_rates(H, Rxx, orders(k, :), Ly, U, N, cb);
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

function Eu_avg = compute_average_energies(Rxx, U, N)
    % compute average energy per user across tones
    Eu_avg = zeros(1, U);

    for u = 1:U
        Eu_avg(u) = sum(Rxx(u, :)) / N;
    end

end

function info = create_info_struct(H, bu_min, w, cb, theta, theta_unique, clusters, ...
        orderings, weights, Rxx_cell, bu_vertices, b_achieved, cvx_status)
    % create comprehensive info structure

    info = struct();
    info.H = H;
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
end
