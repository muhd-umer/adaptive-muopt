function [Rxxs, Eun, w, bun, info] = ada_maxRESMACMIMO(H, Lxu, Etotal, theta, cb, opts)
    %ADA_MAXRESMACMIMO Channel-free adaptive maxRESMAC from pilot samples
    %
    % Learns the optimal transmit design that maximizes the weighted rate sum
    % subject to a single total-energy budget, using only transmitted pilots x0
    % and received samples y = H*Atx*x0 + n. The channel H and noise covariance
    % Rnn are used exclusively by the local sampler that synthesizes y; the
    % learner never forms H, Rnn, Rnn^-1, the channel Gram, or Atx^-1. The only
    % quantity the optimizer reads is the per-stream S0 returned by an adaptive
    % GDFE, giving the weighted rate J as a channel-free oracle. Because J is
    % concave in the transmit covariances (fixed SIC order), projected-gradient
    % ascent on the covariances reaches the same optimum as the canonical solver.
    %
    % Args:
    %   H: channel matrix [Ly, sum(Lxu), N] used by the local sampler only
    %   Lxu: streams per user (scalar for uniform)
    %   Etotal: total energy budget
    %   theta: rate weights
    %   cb: baseband factor, 1 complex or 2 real
    %   opts: optional solver settings
    %
    % Returns:
    %   Rxxs: per-user per-tone transmit covariance matrices
    %   Eun: per-user per-tone energies
    %   w: adaptive common water level estimate (analog of canonical lambda)
    %   bun: per-user per-tone rates
    %   info: diagnostics

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    t_start = tic;
    [Ly, Ltot, N] = size(H);

    if N > 1 && cb == 2
        H = H(:, :, 1:N / 2);
        N = N / 2;
    end

    if N == 1
        H = reshape(H, Ly, [], 1);
    end

    % [subband grouping] solve on a coarser grid of representative tones
    % (one design per subband, applied to all its tones), cutting the feedback
    % and the number of tones that must be probed
    subbands = get_opt(opts, 'subbands', []);

    if ~isempty(subbands) && subbands >= 1 && subbands < N
        H = group_subbands(H, subbands);
        N = size(H, 3);
    end

    Etotal = sum(Etotal(:));
    theta = reshape(theta, [], 1);
    U = length(theta);
    uniform_flag = false;

    if isscalar(Lxu)
        Lxu = ones(1, U) * Lxu;
        uniform_flag = true;
    else
        Lxu = reshape(Lxu, 1, []);
    end

    if sum(Lxu) ~= Ltot
        error('sum(Lxu) = %d must equal total TX antennas = %d', sum(Lxu), Ltot);
    end

    if cb ~= 1 && cb ~= 2
        error('cb must be 1 or 2');
    end

    if Etotal < 0
        error('Etotal must be nonnegative');
    end

    [stheta, order] = sort(theta, 'descend');
    inv_order = zeros(U, 1);

    for u = 1:U
        inv_order(order(u)) = u;
    end

    ranges = user_ranges(Lxu);
    sorted_cols = concatenate_ranges(ranges(order));
    Lxu_sorted = Lxu(order);
    sorted_ranges = user_ranges(Lxu_sorted);
    Ltot_sorted = sum(Lxu_sorted);

    Hs = H(:, sorted_cols, :);

    noise_var = get_opt(opts, 'noiseVar', 1);

    if isfield(opts, 'Rnn') && ~isempty(opts.Rnn)
        Rnn = opts.Rnn;
    else
        Rnn = noise_var * eye(Ly);
    end

    Rnn = 0.5 * (Rnn + Rnn');

    max_outer = get_opt(opts, 'maxOuter', 40);
    Npack = get_opt(opts, 'Nsamp', 8000);
    ridge = get_opt(opts, 'ridge', 1e-9);
    tol_rate = get_opt(opts, 'tolRate', 1e-5);
    probe_step = get_opt(opts, 'probeStep', 1e-6);
    floor_frac = get_opt(opts, 'floorFrac', 2e-3);
    restart_tries = get_opt(opts, 'restartTries', 3);
    iwf_sweeps = get_opt(opts, 'iwfSweeps', 8);
    verbose = get_opt(opts, 'verbose', false);
    rng_seed = get_opt(opts, 'rngSeed', 7);

    % blind (deployable) mode settings
    blind_rounds = get_opt(opts, 'blindRounds', 15);
    probe_pilots = get_opt(opts, 'probePilots', 3000);
    probe_avg = get_opt(opts, 'probeAvg', 4);
    hess_rank = get_opt(opts, 'hessRank', min(Ly, Ltot_sorted));
    ridge_g = get_opt(opts, 'ridgeG', 1e-2);
    ridge_h = get_opt(opts, 'ridgeH', 1e-2);
    probe_batch = get_opt(opts, 'probeBatch', []);
    trust_radius = get_opt(opts, 'trustRadius', []);

    % 'exact' refines the warm start with common-random-number zeroth-order
    % projected-gradient ascent (bit-exact offline reference, not deployable).
    % 'practical' runs the deployable blind surrogate optimizer: batched
    % fresh-noise random probes of the rate oracle, a noise-averaging quadratic
    % surrogate fit, and an analytic trust-region step. Both read only S0
    solve_mode = get_opt(opts, 'mode', 'exact');
    refine = strcmpi(solve_mode, 'exact');
    blind = strcmpi(solve_mode, 'practical') || strcmpi(solve_mode, 'blind');

    rng(rng_seed);

    % frozen pilot/noise block per tone
    Lnn = chol(Rnn +1e-14 * eye(Ly), 'lower');
    Xb = cell(1, N);
    Nb = cell(1, N);

    for n = 1:N
        Xb{n} = (randn(Ltot_sorted, Npack) + 1j * randn(Ltot_sorted, Npack)) / sqrt(2);
        Nb{n} = Lnn * ((randn(Ly, Npack) + 1j * randn(Ly, Npack)) / sqrt(2));
    end

    % precompute fixed pilot/noise moments; each trial precoder's sample
    % covariance is reconstructed from them without re-touching the pilot block
    sam = sampler_moments(Hs, Xb, Nb);
    basis = hermitian_bases(Lxu_sorted);
    Runif = cell(1, N);

    for n = 1:N

        if Etotal <= 0
            Runif{n} = zeros(Ltot_sorted);
        else
            Runif{n} = (Etotal / (Ltot_sorted * N)) * eye(Ltot_sorted);
        end

    end

    % [warm starts] uniform, an iterative-waterfilling design, and opts.R0 if
    % given; each is refined and the best kept
    starts = {Runif};

    if Etotal > 0 && iwf_sweeps > 0
        starts{end + 1} = iwf_init(sam, Runif, sorted_ranges, cb, ridge, stheta, Etotal, floor_frac, iwf_sweeps);
    end

    if isfield(opts, 'R0') && ~isempty(opts.R0)
        starts{end + 1} = project_cov(opts.R0, sorted_ranges, Etotal, floor_frac);
    end

    % resolve blind batch/radius defaults from the surrogate dimension
    m_coords = 0;

    for u = 1:numel(sorted_ranges)
        m_coords = m_coords + numel(basis{u});
    end

    if isempty(probe_batch)
        % ~2.5x oversample of the quadratic-surrogate parameter count so the
        % per-tone curvature is well identified, capped for runtime
        nparam = 1 + 2 * m_coords + m_coords * (m_coords - 1) / 2;
        probe_batch = min(round(2.5 * nparam), 250);
    end

    if isempty(trust_radius)
        trust_radius = 0.15 * Etotal / max(N * Ltot_sorted, 1);
    end

    best_rate = -inf;
    Rblk = starts{1};
    last = evaluate_cov(sam, Rblk, sorted_ranges, cb, ridge);
    rate_hist = [];
    iters = 0;
    converged = false;

    for si = 1:numel(starts)

        if refine
            [R_s, last_s, rate_s, it_s, conv_s, hist_s] = run_pg(sam, starts{si}, sorted_ranges, cb, ...
                ridge, stheta, Etotal, basis, probe_step, floor_frac, max_outer, tol_rate, ...
                restart_tries, verbose);
        elseif blind
            % the uniform start carries no direction information; refine only
            % the informed warm start(s) when one is available
            if si == 1 && numel(starts) > 1
                continue;
            end

            [R_s, last_s, rate_s, it_s, conv_s, hist_s] = run_blind(Hs, Lnn, Ltot_sorted, Ly, ...
                starts{si}, sorted_ranges, cb, ridge, stheta, Etotal, basis, floor_frac, ...
                blind_rounds, probe_pilots, probe_avg, probe_batch, trust_radius, hess_rank, ...
                ridge_g, ridge_h, verbose);
        else
            R_s = starts{si};
            last_s = evaluate_cov(sam, R_s, sorted_ranges, cb, ridge, true);
            rate_s = stheta.' * sum(last_s.bun, 2);
            it_s = 0;
            conv_s = true;
            hist_s = rate_s;
        end

        iters = iters + it_s;

        if rate_s > best_rate
            best_rate = rate_s;
            Rblk = R_s;
            last = last_s;
            rate_hist = hist_s;
            converged = conv_s;
        end

    end

    % re-measure the best design with the full GDFE for the W/B/Fhat outputs
    last = evaluate_cov(sam, Rblk, sorted_ranges, cb, ridge, true);

    Rxx_cell = cell(U, N);
    Eun = zeros(U, N);
    bun = zeros(U, N);
    S0_out = zeros(Ltot_sorted, N);
    Mhat = zeros(Ltot_sorted, Ltot_sorted, N);

    for su = 1:U
        idx = sorted_ranges{su};
        u = order(su);

        for n = 1:N
            R = 0.5 * (Rblk{n}(idx, idx) + Rblk{n}(idx, idx)');
            Rxx_cell{u, n} = R;
            Eun(u, n) = real(trace(R));
            bun(u, n) = last.bun(su, n);
            [V, ~] = eig(R);
            Mhat(idx, idx, n) = V;
        end

        S0_out(idx, :) = last.S0(idx, :);
    end

    Lxu_max = max(Lxu);

    if uniform_flag
        Rxxs = zeros(Lxu_max, Lxu_max, U, N);

        for u = 1:U

            for n = 1:N
                Rxxs(1:Lxu(u), 1:Lxu(u), u, n) = Rxx_cell{u, n};
            end

        end

    else
        Rxxs = Rxx_cell;
    end

    lambda = common_level_from_cov(last.S0, Rblk, stheta, sorted_ranges);
    w = lambda * ones(U, 1);

    info = struct();
    info.mode = solve_mode;
    info.S0 = unsort_stream_tone(S0_out, order, Lxu, N);
    info.S0_sorted = S0_out;
    info.order = order;
    info.W = last.W;
    info.B = last.B;
    info.Mhat = Mhat;
    info.weighted_rate = theta.' * sum(bun, 2);
    info.rate_hist = rate_hist;
    info.iters = iters;
    info.converged = converged;
    info.pilots = Npack;

    if blind
        info.probeBatch = probe_batch;
        info.probePilots = probe_pilots;
        info.probeRounds = iters;
        info.probes = iters * probe_batch;
    end

    info.runtime = toc(t_start);
    info.opts = opts;
end

function [Rblk, last, best_rate, iters, converged, rate_hist] = run_pg(sam, Rblk, ranges, cb, ridge, ...
        theta, total, basis, probe_step, floor_frac, max_outer, tol_rate, restart_tries, verbose)
    % projected-gradient ascent from a fixed start; tracks the best design and
    % restarts on stall
    rate_hist = zeros(1, max_outer);
    last = evaluate_cov(sam, Rblk, ranges, cb, ridge);
    converged = false;
    iters = 0;
    prev_rate = -inf;
    best_rate = theta.' * sum(last.bun, 2);
    best_R = Rblk;
    best_last = last;
    stalls = 0;

    for iter = 1:max_outer
        iters = iter;
        weighted_rate = theta.' * sum(last.bun, 2);
        rate_hist(iter) = weighted_rate;

        if weighted_rate > best_rate
            best_rate = weighted_rate;
            best_R = Rblk;
            best_last = last;
        end

        if verbose
            fprintf('ada maxresmac iter %d/%d weighted_rate %.8g\n', iter, max_outer, weighted_rate);
        end

        if iter > 1 && abs(weighted_rate - prev_rate) / max(1, abs(prev_rate)) < tol_rate && stalls == 0
            converged = true;
            break;
        end

        prev_rate = weighted_rate;

        if total > 0
            [Rblk, last, moved] = rate_grad_step(sam, Rblk, ranges, cb, ridge, ...
                theta, total, basis, probe_step, floor_frac, last);

            if ~moved

                if stalls >= restart_tries
                    converged = true;
                    break;
                end

                stalls = stalls + 1;
                Rblk = perturb_cov(best_R, ranges, total, floor_frac, 0.25 * 0.5 ^ (stalls - 1));
                last = evaluate_cov(sam, Rblk, ranges, cb, ridge);
            else
                stalls = 0;
            end

        end

    end

    Rblk = best_R;
    last = best_last;
    rate_hist = rate_hist(1:max(iters, 1));
end

function [Rblk, last, best_rate, iters, converged, rate_hist] = run_blind(Hs, Lnn, Ltot, Ly, ...
        Rblk, ranges, cb, ridge, theta, total, basis, floor_frac, rounds, probe_pilots, ...
        probe_avg, batch, trust_radius, hess_rank, ridge_g, ridge_h, verbose)
    % deployable blind surrogate optimizer. Around the current design, transmit a
    % batch of random probe designs with independent (non-frozen) noise, measure
    % their weighted rate with the channel-free S0 oracle, fit a concave quadratic
    % surrogate per tone by noise-averaging least squares, and take an analytic
    % trust-region step. No common random numbers, so the estimated gradient
    % survives real over-the-air noise (its variance decays as 1/(batch*rho^2)
    % rather than blowing up like a vanishing-step finite difference); the rate
    % is concave in the covariances, so the iteration reaches the global optimum
    % up to estimation error
    N = numel(Rblk);
    m = 0;

    for u = 1:numel(ranges)
        m = m + numel(basis{u});
    end

    if isempty(batch)
        nparam = 1 + 2 * m + m * (m - 1) / 2;
        batch = min(round(2.5 * nparam), 250);
    end

    if isempty(trust_radius)
        trust_radius = 0.15 * total / max(N * Ltot, 1);
    end

    rho = trust_radius;
    rho_max = 4 * trust_radius;
    nmode = Ltot * N;
    efloor = min(total * floor_frac / max(nmode, 1), total / max(2 * nmode, 1));
    rho_min = 3 * efloor;
    resets = 0;
    max_resets = 2;

    % fixed known pilot block (transmitter-controlled, reused across probes); only
    % the noise is redrawn per probe, so Pxx is computed once here
    Xb_fix = cell(1, N);
    Pxx_fix = cell(1, N);

    for n = 1:N
        Xb_fix{n} = (randn(Ltot, probe_pilots) + 1j * randn(Ltot, probe_pilots)) / sqrt(2);
        Pxx_fix{n} = (Xb_fix{n} * Xb_fix{n}') / probe_pilots;
    end

    % accept/reject and best-tracking use a rate averaged over several fresh
    % blocks; a single noisy measurement would make those decisions wander
    cur_rate = meas_rate(Hs, Xb_fix, Pxx_fix, Ly, Lnn, probe_avg, Rblk, ranges, cb, ridge, theta);
    best_rate = cur_rate;
    best_R = Rblk;
    rate_hist = zeros(1, rounds);
    iters = 0;
    converged = false;

    for r = 1:rounds
        iters = r;

        % batch of fresh-noise random probes; each probe is one design measured
        % across all tones at once, so the probe count is independent of N
        Cn = cell(1, N);
        Yn = cell(1, N);

        for n = 1:N
            Cn{n} = zeros(batch, m);
            Yn{n} = zeros(batch, 1);
        end

        for i = 1:batch
            Rcand = cell(1, N);

            for n = 1:N
                D = coords_to_herm(rho * randn(m, 1), basis, ranges, Ltot);
                Rcand{n} = Rblk{n} + D;
            end

            Rcand = project_cov(Rcand, ranges, total, floor_frac);
            sam_i = sample_fresh(Hs, Xb_fix, Pxx_fix, Ly, Lnn);
            res = evaluate_cov(sam_i, Rcand, ranges, cb, ridge);

            for n = 1:N
                % regress against the projected probe, not the nominal step
                Cn{n}(i, :) = herm_to_coords(Rcand{n} - Rblk{n}, basis, ranges).';
                Yn{n}(i) = theta.' * res.bun(:, n);
            end

        end

        % per-tone surrogate fit and trust-region step (shared probe directions,
        % per-tone steps from per-tone curvature)
        Rnew = cell(1, N);
        pred_gain = 0;

        for n = 1:N
            [g, Hn] = fit_surrogate(Cn{n}, Yn{n}, m, ridge_g, ridge_h, hess_rank);
            Mc = -Hn;
            c = tr_step(g, Mc, rho * sqrt(m));
            Rnew{n} = Rblk{n} + coords_to_herm(c, basis, ranges, Ltot);
            pred_gain = pred_gain + g.' * c - 0.5 * real(c.' * Mc * c);
        end

        Rnew = project_cov(Rnew, ranges, total, floor_frac);
        rate_a = meas_rate(Hs, Xb_fix, Pxx_fix, Ly, Lnn, probe_avg, Rnew, ranges, cb, ridge, theta);

        % also consider an analytic energy reset along the updated directions and
        % keep whichever is better: the reset wins when the warm-start energy
        % allocation is already good, free probing wins when it needs refining
        Rreset = energy_reset(sample_fresh(Hs, Xb_fix, Pxx_fix, Ly, Lnn), Rnew, ranges, cb, ridge, theta, total, floor_frac);
        rate_b = meas_rate(Hs, Xb_fix, Pxx_fix, Ly, Lnn, probe_avg, Rreset, ranges, cb, ridge, theta);

        if rate_b > rate_a
            Rnew = Rreset;
            new_rate = rate_b;
        else
            new_rate = rate_a;
        end

        kappa = (new_rate - cur_rate) / max(pred_gain, realmin);

        if new_rate > cur_rate
            Rblk = Rnew;
            cur_rate = new_rate;

            if kappa > 0.75
                rho = min(1.4 * rho, rho_max);
            elseif kappa < 0.1
                rho = 0.6 * rho;
            end

        else
            rho = 0.6 * rho;
        end

        if cur_rate > best_rate
            best_rate = cur_rate;
            best_R = Rblk;
        end

        rate_hist(r) = cur_rate;

        if verbose
            fprintf('ada blind round %d/%d rate %.8g rho %.3g kappa %.3g\n', r, rounds, cur_rate, rho, kappa);
        end

        if rho <= rho_min
            % re-expand a few times to escape a noise-induced collapse before
            % declaring convergence; best_R is retained across resets
            if resets < max_resets
                resets = resets + 1;
                rho = 0.5 * trust_radius;
                Rblk = best_R;
                cur_rate = best_rate;
            else
                converged = true;
                break;
            end

        end

    end

    Rblk = best_R;
    last = evaluate_cov(sample_fresh(Hs, Xb_fix, Pxx_fix, Ly, Lnn), best_R, ranges, cb, ridge, true);
    rate_hist = rate_hist(1:max(iters, 1));
end

function rate = meas_rate(Hs, Xb, Pxx, Ly, Lnn, K, Rblk, ranges, cb, ridge, theta)
    % weighted rate averaged over K independent fresh-noise blocks, to denoise
    % the accept/reject and best-tracking decisions
    acc = 0;

    for k = 1:max(K, 1)
        sam = sample_fresh(Hs, Xb, Pxx, Ly, Lnn);
        res = evaluate_cov(sam, Rblk, ranges, cb, ridge);
        acc = acc + theta.' * sum(res.bun, 2);
    end

    rate = acc / max(K, 1);
end

function sam = sample_fresh(Hs, Xb, Pxx, Ly, Lnn)
    % redraw only the noise (fresh and independent per probe) against a fixed,
    % known pilot block, and return the joint sample-covariance moments. Pilots
    % are transmitter-controlled and reused, so Pxx is precomputed once; only the
    % noise-dependent moments are formed here. H/Rnn appear only here and in
    % reconstruct_S; the learner still consumes only the reconstructed S
    N = size(Hs, 3);
    Npack = size(Xb{1}, 2);
    sam = struct('H', cell(1, N), 'Pxx', cell(1, N), 'Pxn', cell(1, N), 'Pnn', cell(1, N));

    for n = 1:N
        Nb = Lnn * ((randn(Ly, Npack) + 1j * randn(Ly, Npack)) / sqrt(2));
        sam(n).H = Hs(:, :, n);
        sam(n).Pxx = Pxx{n};
        sam(n).Pxn = (Xb{n} * Nb') / Npack;
        sam(n).Pnn = (Nb * Nb') / Npack;
    end

end

function [g, H] = fit_surrogate(C, y, m, ridge_g, ridge_h, hess_rank)
    % fit Jhat(c) = a + g'c + 1/2 c'H c by ridge least squares over the batch,
    % with features [1, c_i, c_i^2, c_i c_j (i<j)]. Columns are scaled to unit
    % norm before ridging so the penalties are dimensionless, then unscaled.
    % H is symmetrized and projected to negative semidefinite (rank <= hess_rank)
    % so the trust-region subproblem is a bounded concave maximization
    B = size(C, 1);
    npair = m * (m - 1) / 2;
    P = 1 + 2 * m + npair;
    Phi = ones(B, P);
    Phi(:, 2:m + 1) = C;
    Phi(:, m + 2:2 * m + 1) = C .^ 2;
    pair_idx = zeros(npair, 2);
    col = 2 * m + 2;
    k = 0;

    for i = 1:m

        for j = i + 1:m
            k = k + 1;
            Phi(:, col) = C(:, i) .* C(:, j);
            pair_idx(k, :) = [i, j];
            col = col + 1;
        end

    end

    scale = ones(1, P);

    for c = 2:P
        s = norm(Phi(:, c));

        if s > 0
            scale(c) = s;
        end

    end

    Phis = Phi ./ scale;
    pen = zeros(1, P);
    pen(2:m + 1) = ridge_g;
    pen(m + 2:end) = ridge_h;
    beta = (Phis.' * Phis + diag(pen)) \ (Phis.' * y);
    beta = beta ./ scale.';

    g = beta(2:m + 1);
    diagq = beta(m + 2:2 * m + 1);
    H = zeros(m);

    for i = 1:m
        H(i, i) = 2 * diagq(i);
    end

    for k = 1:npair
        i = pair_idx(k, 1);
        j = pair_idx(k, 2);
        H(i, j) = beta(2 * m + 1 + k);
        H(j, i) = H(i, j);
    end

    H = 0.5 * (H + H');
    [V, D] = eig(H);
    d = real(diag(D));
    d(d > 0) = 0;
    [~, o] = sort(d, 'ascend');
    keep = false(m, 1);
    keep(o(1:min(max(hess_rank, 0), m))) = true;
    d(~keep) = 0;
    H = V * diag(d) * V';
    H = 0.5 * (H + H');
end

function c = tr_step(g, M, radius)
    % trust-region step: maximize g'c - 1/2 c'M c over ||c|| <= radius, M >= 0.
    % Levenberg damping c(tau) = (M + tau I)^{-1} g with the smallest tau >= 0
    % that meets the radius (||c(tau)|| is decreasing in tau); handles M = 0
    % (pure gradient step to the boundary) and g = 0
    m = numel(g);

    if norm(g) == 0
        c = zeros(m, 1);
        return;
    end

    M = 0.5 * (M + M');
    I = eye(m);
    lam_min = min(real(eig(M)));

    if lam_min > 1e-12
        c0 = M \ g;

        if norm(c0) <= radius
            c = c0;
            return;
        end

    end

    hi = 1;

    while norm((M + hi * I) \ g) > radius && hi < 1e12
        hi = hi * 10;
    end

    lo = 0;

    for k = 1:60
        tau = 0.5 * (lo + hi);

        if norm((M + tau * I) \ g) > radius
            lo = tau;
        else
            hi = tau;
        end

    end

    c = (M + hi * I) \ g;
end

function c = herm_to_coords(Delta, basis, ranges)
    % coordinates of a block-diagonal Hermitian Delta in the orthonormal basis
    c = [];

    for u = 1:numel(ranges)
        idx = ranges{u};
        blk = Delta(idx, idx);
        nb = numel(basis{u});
        cu = zeros(nb, 1);

        for b = 1:nb
            cu(b) = real(trace(basis{u}{b}' * blk));
        end

        c = [c; cu];
    end

end

function D = coords_to_herm(c, basis, ranges, Ltot)
    % inverse of herm_to_coords: assemble the block-diagonal Hermitian matrix
    D = zeros(Ltot);
    off = 0;

    for u = 1:numel(ranges)
        idx = ranges{u};
        nb = numel(basis{u});
        blk = zeros(numel(idx));

        for b = 1:nb
            blk = blk + c(off + b) * basis{u}{b};
        end

        off = off + nb;
        D(idx, idx) = blk;
    end

    D = 0.5 * (D + D');
end

function result = evaluate_cov(sam, Rblk, ranges, cb, ridge, needW)
    % needW = true returns the full GDFE (W, B, Fhat); false takes the fast
    % S0-only path used by the ascent, which never reads W/B/Fhat
    if nargin < 6
        needW = false;
    end

    N = numel(sam);
    Ltot = size(sam(1).Pxx, 1);
    Ly = size(sam(1).H, 1);
    U = numel(ranges);
    result.S0 = ones(Ltot, N);
    result.bun = zeros(U, N);
    result.W = cell(1, N);
    result.B = cell(1, N);
    result.Fhat = cell(1, N);

    for n = 1:N
        Atx = cov_sqrt(Rblk{n}, ranges);
        [S, Fhat] = reconstruct_S(sam(n), Atx);

        if needW
            [W, B, ~, S0] = gdfe_from_S(S, Ltot, Ly, ridge);
            result.W{n} = W;
            result.B{n} = B;
            result.Fhat{n} = Fhat;
        else
            S0 = s0_from_S(S, Ltot, Ly, ridge);
        end

        result.S0(:, n) = S0;

        for u = 1:U
            idx = ranges{u};
            result.bun(u, n) = sum(log2(max(S0(idx), 1))) / cb;
        end

    end

end

function bun_tone = evaluate_atx_tone(sam_n, Atx, ranges, cb, ridge)
    U = numel(ranges);
    Ltot = size(sam_n.Pxx, 1);
    Ly = size(sam_n.H, 1);
    S = reconstruct_S(sam_n, Atx);
    S0 = s0_from_S(S, Ltot, Ly, ridge);
    bun_tone = zeros(U, 1);

    for u = 1:U
        idx = ranges{u};
        bun_tone(u) = sum(log2(max(S0(idx), 1))) / cb;
    end

end

function Atx = cov_sqrt(Rblk, ranges)
    Atx = zeros(size(Rblk));

    for u = 1:numel(ranges)
        idx = ranges{u};
        Atx(idx, idx) = sqrtm_psd(Rblk(idx, idx));
    end

end

function [Rblk, last, moved] = rate_grad_step(sam, Rblk, ranges, cb, ridge, theta, total, basis, probe_step, floor_frac, last)
    % one zeroth-order gradient step on the measured weighted rate; probe the
    % rate oracle along an orthonormal Hermitian basis of each user block to
    % form the covariance gradient, then project the step back onto the feasible
    % set. Uses only the channel-free S0 oracle
    N = numel(sam);
    Ltot = size(sam(1).Pxx, 1);
    U = numel(ranges);
    base = theta.' * sum(last.bun, 2);
    h = probe_step * max(total, 1);
    moved = false;

    % each probe perturbs one user block, so cache the base block square-root
    % and recompute only the perturbed block
    Gamma = cell(1, N);

    for n = 1:N
        Gamma{n} = zeros(Ltot);
        base_n = theta.' * last.bun(:, n);
        Atx_base = cov_sqrt(Rblk{n}, ranges);

        for u = 1:U
            idx = ranges{u};

            for b = 1:numel(basis{u})
                D = basis{u}{b};
                Atx = Atx_base;
                Atx(idx, idx) = sqrtm_psd(Rblk{n}(idx, idx) + h * D);
                bt = evaluate_atx_tone(sam(n), Atx, ranges, cb, ridge);
                g = (theta.' * bt - base_n) / h;
                Gamma{n}(idx, idx) = Gamma{n}(idx, idx) + g * D;
            end

        end

    end

    gnorm = 0;

    for n = 1:N
        gnorm = gnorm + norm(Gamma{n}, 'fro') ^ 2;
    end

    gnorm = sqrt(gnorm);

    if gnorm <= 0 || ~isfinite(gnorm)
        return;
    end

    % normalized direction with a wide geometric step ladder, so the step can
    % shrink enough to keep improving near the optimum
    for n = 1:N
        Gamma{n} = Gamma{n} / gnorm;
    end

    best_rate = base;
    declines = 0;
    alphas = total * 2 .^ (0:-1:-24);

    for ai = 1:numel(alphas)
        alpha = alphas(ai);
        Rcand = cell(1, N);

        for n = 1:N
            Rcand{n} = Rblk{n} + alpha * Gamma{n};
        end

        Rcand = project_cov(Rcand, ranges, total, floor_frac);
        trial = evaluate_cov(sam, Rcand, ranges, cb, ridge);
        cand_rate = theta.' * sum(trial.bun, 2);

        if cand_rate > best_rate +1e-9
            Rblk = Rcand;
            last = trial;
            best_rate = cand_rate;
            moved = true;
            declines = 0;
        elseif moved
            % scanning large->small alpha; once an improving step is found, the
            % shrinking-step tail decays back toward base, so stop after the peak
            declines = declines + 1;

            if declines >= 2
                break;
            end

        end

    end

end

function Rblk = iwf_init(sam, Rblk, ranges, cb, ridge, theta, total, floor_frac, sweeps)
    % [iterative water-filling] warm start; per user block the interference-
    % whitened channel Gram Ge_int is recovered channel-free from the learned
    % gain Kbar = W_u*F_u ~ Atx_u'*Ge*Atx_u by de-energizing with the known
    % R_u^{1/2}, so its eigenvectors are the beam directions and a global weighted
    % waterfilling over all mode gains sets the energies at a common water level
    N = numel(Rblk);
    U = numel(ranges);

    for s = 1:sweeps
        last = evaluate_cov(sam, Rblk, ranges, cb, ridge, true);
        Vc = cell(N, U);
        gains = cell(N, U);
        pooled_g = [];
        pooled_w = [];

        for n = 1:N

            for u = 1:U
                idx = ranges{u};
                L = numel(idx);
                R = 0.5 * (Rblk{n}(idx, idx) + Rblk{n}(idx, idx)');
                [Vr, Dr] = eig(R);
                dr = max(real(diag(Dr)), total * floor_frac / (N * U * L));
                Rinv2 = Vr * diag(1 ./ sqrt(dr)) * Vr';
                Kbar = 0.5 * (last.W{n}(idx, :) * last.Fhat{n}(:, idx) + (last.W{n}(idx, :) * last.Fhat{n}(:, idx))');
                [Vk, Dk] = eig(Kbar);
                dk = min(max(real(diag(Dk)), 0), 1 -1e-9);
                T = Vk * diag(dk ./ (1 - dk)) * Vk';
                Ge = Rinv2 * (0.5 * (T + T')) * Rinv2;
                Ge = 0.5 * (Ge + Ge');
                [V, D] = eig(Ge);
                g = max(real(diag(D)), 0);
                [g, o] = sort(g, 'descend');
                Vc{n, u} = V(:, o);
                gains{n, u} = g;
                pooled_g = [pooled_g; g];
                pooled_w = [pooled_w; theta(u) * ones(L, 1)];
            end

        end

        lam = waterfill_level(pooled_g, pooled_w, total);

        for n = 1:N

            for u = 1:U
                idx = ranges{u};
                g = gains{n, u};
                lev = max(theta(u) / lam - 1 ./ max(g, realmin), 0);
                V = Vc{n, u};
                Rblk{n}(idx, idx) = V * diag(lev) * V';
            end

        end

        Rblk = project_cov(Rblk, ranges, total, floor_frac);
    end

end

function Rblk = energy_reset(sam, Rblk, ranges, cb, ridge, theta, total, floor_frac)
    % analytic waterfilling that keeps each block's current eigenvectors and only
    % re-sets the eigenvalues from the channel-free interference-whitened gain
    % along those directions. Same learned quantities as iwf_init (W, Fhat), so
    % it stays inside the non-cheating contract
    N = numel(Rblk);
    U = numel(ranges);
    last = evaluate_cov(sam, Rblk, ranges, cb, ridge, true);
    Vc = cell(N, U);
    gains = cell(N, U);
    pooled_g = [];
    pooled_w = [];

    for n = 1:N

        for u = 1:U
            idx = ranges{u};
            L = numel(idx);
            R = 0.5 * (Rblk{n}(idx, idx) + Rblk{n}(idx, idx)');
            [Vr, Dr] = eig(R);
            dr = max(real(diag(Dr)), total * floor_frac / (N * U * L));
            Rinv2 = Vr * diag(1 ./ sqrt(dr)) * Vr';
            Kbar = 0.5 * (last.W{n}(idx, :) * last.Fhat{n}(:, idx) + (last.W{n}(idx, :) * last.Fhat{n}(:, idx))');
            [Vk, Dk] = eig(Kbar);
            dk = min(max(real(diag(Dk)), 0), 1 -1e-9);
            T = Vk * diag(dk ./ (1 - dk)) * Vk';
            Ge = Rinv2 * (0.5 * (T + T')) * Rinv2;
            Ge = 0.5 * (Ge + Ge');
            g = max(real(diag(Vr' * Ge * Vr)), 0);
            Vc{n, u} = Vr;
            gains{n, u} = g;
            pooled_g = [pooled_g; g];
            pooled_w = [pooled_w; theta(u) * ones(L, 1)];
        end

    end

    lam = waterfill_level(pooled_g, pooled_w, total);

    for n = 1:N

        for u = 1:U
            idx = ranges{u};
            g = gains{n, u};
            lev = max(theta(u) / lam - 1 ./ max(g, realmin), 0);
            V = Vc{n, u};
            Rblk{n}(idx, idx) = V * diag(lev) * V';
        end

    end

    Rblk = project_cov(Rblk, ranges, total, floor_frac);
end

function lam = waterfill_level(gains, weights, total)
    active = gains > 0 & weights > 0;

    if ~any(active) || total <= 0
        lam = 1;
        return;
    end

    lo = realmin;
    hi = max(weights(active) .* gains(active)) * 2 + 1;

    for k = 1:200
        lam = sqrt(lo * hi);
        lev = max(weights(active) / lam - 1 ./ gains(active), 0);

        if sum(lev) > total
            lo = lam;
        else
            hi = lam;
        end

    end

    lam = sqrt(lo * hi);
end

function Rblk = perturb_cov(Rblk, ranges, total, floor_frac, sigma)
    % random feasible perturbation: add scaled Hermitian noise per user block
    N = numel(Rblk);

    for n = 1:N

        for u = 1:numel(ranges)
            idx = ranges{u};
            L = numel(idx);
            G = (randn(L) + 1j * randn(L)) / sqrt(2);
            G = 0.5 * (G + G');
            scale = sigma * (total / (N * numel(ranges)));
            Rblk{n}(idx, idx) = Rblk{n}(idx, idx) + scale * G;
        end

    end

    Rblk = project_cov(Rblk, ranges, total, floor_frac);
end

function Rblk = project_cov(Rblk, ranges, total, floor_frac)
    % Euclidean projection onto {R_u >= 0 block-diagonal, sum_{u,n} tr(R_u) =
    % total}. Each block shares its eigenvectors; the pooled eigenvalues are
    % projected jointly onto the nonnegative simplex of mass total
    N = numel(Rblk);
    U = numel(ranges);

    if total <= 0

        for n = 1:N
            Rblk{n} = zeros(size(Rblk{n}));
        end

        return;
    end

    Vs = cell(N, U);
    evs = cell(N, U);
    pooled = [];

    for n = 1:N

        for u = 1:U
            idx = ranges{u};
            R = 0.5 * (Rblk{n}(idx, idx) + Rblk{n}(idx, idx)');

            if isscalar(R)
                V = 1;
                e = real(R);
            else
                [V, D] = eig(R);
                e = real(diag(D));
            end

            Vs{n, u} = V;
            evs{n, u} = e;
            pooled = [pooled; e];
        end

    end

    nmode = numel(pooled);
    efloor = min(total * floor_frac / max(nmode, 1), total / max(2 * nmode, 1));
    pooled = efloor + project_simplex_vector(pooled - efloor, total - efloor * nmode);

    off = 0;

    for n = 1:N

        for u = 1:U
            idx = ranges{u};
            k = numel(evs{n, u});
            e = pooled(off + (1:k));
            off = off + k;
            V = Vs{n, u};

            if isscalar(e)
                R = e;
            else
                R = V * diag(e) * V';
            end

            Rblk{n}(idx, idx) = 0.5 * (R + R');
        end

    end

end

function x = project_simplex_vector(v, total)
    v = v(:);

    if total <= 0
        x = zeros(size(v));
        return;
    end

    u = sort(v, 'descend');
    cssv = cumsum(u) - total;
    rho = find(u - cssv ./ (1:numel(u))' > 0, 1, 'last');

    if isempty(rho)
        x = total * ones(size(v)) / numel(v);
        return;
    end

    tau = cssv(rho) / rho;
    x = max(v - tau, 0);
end

function basis = hermitian_bases(Lxu)
    % orthonormal Hermitian basis per user block under the real trace inner
    % product <X,Y> = real(trace(X'*Y))
    basis = cell(1, numel(Lxu));

    for u = 1:numel(Lxu)
        L = Lxu(u);
        blk = cell(1, L * L);
        c = 0;

        for i = 1:L
            D = zeros(L);
            D(i, i) = 1;
            c = c + 1;
            blk{c} = D;
        end

        for i = 1:L

            for j = i + 1:L
                S = zeros(L);
                S(i, j) = 1 / sqrt(2);
                S(j, i) = 1 / sqrt(2);
                c = c + 1;
                blk{c} = S;

                A = zeros(L);
                A(i, j) = 1j / sqrt(2);
                A(j, i) = -1j / sqrt(2);
                c = c + 1;
                blk{c} = A;
            end

        end

        basis{u} = blk(1:c);
    end

end

function sam = sampler_moments(H, Xb, Nb)
    % precompute the fixed second moments of the frozen pilot and noise blocks
    % per tone. H, Xb, Nb are used ONLY here (for  simulation); the
    % learner downstream consumes only the reconstructed sample covariance S
    N = numel(Xb);
    sam = struct('H', cell(1, N), 'Pxx', cell(1, N), 'Pxn', cell(1, N), 'Pnn', cell(1, N));

    for n = 1:N
        Npack = size(Xb{n}, 2);
        sam(n).H = H(:, :, n);
        sam(n).Pxx = (Xb{n} * Xb{n}') / Npack;
        sam(n).Pxn = (Xb{n} * Nb{n}') / Npack;
        sam(n).Pnn = (Nb{n} * Nb{n}') / Npack;
    end

end

function [S, Fhat] = reconstruct_S(sam_n, Atx)
    % reconstruct the joint sample covariance S = [Y;X][Y;X]'/Npack for the
    % trial precoder Atx without touching the pilot block, using the fixed
    % moments and the effective response Gx = H*Atx. This is algebraically the
    % same S the learner would form from (X,Y) with Y = Gx*X + N
    Gx = sam_n.H * Atx;
    GP = Gx * sam_n.Pxx;
    Syx = GP + sam_n.Pxn';
    Syy = GP * Gx' + Gx * sam_n.Pxn + sam_n.Pxn' * Gx' + sam_n.Pnn;
    Syy = 0.5 * (Syy + Syy');
    Sxx = 0.5 * (sam_n.Pxx + sam_n.Pxx');
    S = [Syy, Syx; Syx', Sxx];
    S = 0.5 * (S + S');
    Fhat = Syx;
end

function S0 = s0_from_S(S, Ltot, Ly, ridge)
    % per-stream S0 = 1/mse read from a single Cholesky of the joint sample
    % covariance reordered as [Y, X(Ltot), ..., X(1)]. For a lower Cholesky
    % C = L*L', the pivot L(k,k)^2 is the conditional variance of variable k
    % given all earlier variables, which is exactly the GDFE per-stream mse
    % under the SIC order (X(ii) conditioned on Y and the later streams). This
    % replaces the per-stream regression loop with one factorization
    perm = [1:Ly, Ly + (Ltot:-1:1)];
    C = S(perm, perm);
    C = 0.5 * (C + C') + ridge * eye(Ly + Ltot);
    [L, flag] = chol(C, 'lower');

    if flag ~= 0
        [~, ~, ~, S0] = gdfe_from_S(S, Ltot, Ly, ridge);
        return;
    end

    d = real(diag(L)) .^ 2;
    mse = d(Ly + (Ltot:-1:1));
    S0 = max(1, 1 ./ max(mse, realmin));
end

function [W, B, mse, S0] = gdfe_from_S(S, Ltot, Ly, ridge)
    % per-stream MMSE GDFE read off the joint sample covariance S (rows 1:Ly are
    % Y, rows Ly+1:end are X): feedforward W, strictly upper feedback B,
    % per-stream mse and S0=1/mse. Submatrix selection only, no channel formed
    Bmask = triu(true(Ltot), 1);
    W = zeros(Ltot, Ly);
    B = zeros(Ltot, Ltot);
    mse = zeros(Ltot, 1);
    S0 = ones(Ltot, 1);

    for ii = 1:Ltot
        fb = find(Bmask(ii, :));
        sel = [1:Ly, Ly + fb];
        Rphi = S(sel, sel);
        pvec = S(sel, Ly + ii);
        hrow = ((Rphi + ridge * eye(size(Rphi))) \ pvec)';
        W(ii, :) = hrow(1:Ly);
        B(ii, fb) = hrow(Ly + 1:end);
        m = real(S(Ly + ii, Ly + ii) - 2 * real(hrow * pvec) + hrow * Rphi * hrow');
        mse(ii) = max(m, realmin);
        S0(ii) = max(1, 1 / mse(ii));
    end

    B(~Bmask) = 0;
end

function lambda = common_level_from_cov(S0, Rblk, theta, ranges)
    U = numel(theta);
    N = numel(Rblk);
    vals = [];

    for n = 1:N

        for u = 1:U
            idx = ranges{u};
            R = 0.5 * (Rblk{n}(idx, idx) + Rblk{n}(idx, idx)');
            [V, D] = eig(R);
            e = real(diag(D));
            s0 = real(diag(V' * diag(max(real(S0(idx, n)), 1)) * V));
            act = e > max(e) * 1e-6 & e > 0;
            g = theta(u) * (s0(act) - 1) ./ max(e(act) .* s0(act), realmin);
            vals = [vals; g(:)];
        end

    end

    vals = vals(isfinite(vals) & vals > 0);

    if isempty(vals)
        lambda = 0;
    else
        lambda = median(vals);
    end

end

function out = unsort_stream_tone(S0s, order, Lxu, N)
    ranges = user_ranges(Lxu);
    sorted_ranges = user_ranges(Lxu(order));
    out = zeros(sum(Lxu), N);

    for u = 1:numel(Lxu)
        su = find(order == u, 1);
        out(ranges{u}, :) = S0s(sorted_ranges{su}, :);
    end

end

function ranges = user_ranges(Lxu)
    ranges = cell(1, numel(Lxu));
    first = 1;

    for u = 1:numel(Lxu)
        last = first + Lxu(u) - 1;
        ranges{u} = first:last;
        first = last + 1;
    end

end

function cols = concatenate_ranges(ranges)
    cols = [];

    for u = 1:numel(ranges)
        cols = [cols ranges{u}];
    end

end

function Hsb = group_subbands(H, Nsb)
    % collapse N tones into Nsb contiguous subbands, each represented by its
    % center tone (channel assumed flat within a subband)
    N = size(H, 3);
    edges = round(linspace(0, N, Nsb + 1));
    Hsb = zeros(size(H, 1), size(H, 2), Nsb);

    for s = 1:Nsb
        idx = (edges(s) + 1):edges(s + 1);
        Hsb(:, :, s) = H(:, :, idx(ceil(numel(idx) / 2)));
    end

end

function A = project_psd(A)
    A = 0.5 * (A + A');
    [V, D] = eig(A);
    d = max(real(diag(D)), 0);
    A = V * diag(d) * V';
    A = 0.5 * (A + A');
end

function S = sqrtm_psd(A)

    if isscalar(A)
        S = sqrt(max(real(A), 0));
        return;
    end

    A = 0.5 * (A + A');
    [V, D] = eig(A);
    S = V * diag(sqrt(max(real(diag(D)), 0))) * V';
    S = 0.5 * (S + S');
end

function val = get_opt(opts, name, default)

    if isfield(opts, name)
        val = opts.(name);
    else
        val = default;
    end

end
