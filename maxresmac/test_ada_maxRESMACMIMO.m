%TEST_ADA_MAXRESMACMIMO Tests for ada_maxRESMACMIMO

clear;
clc;
rng(42);

tests = struct();

tests(1).name = 'basic mimo';
H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(1).H = H;
tests(1).Lxu = [2 2];
tests(1).Etotal = 10;
tests(1).theta = [1; 1];
tests(1).cb = 1;
tests(1).Nsamp = 1024;

tests(2).name = 'mimo unequal antennas';
tests(2).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(2).Lxu = [2 3];
tests(2).Etotal = 8;
tests(2).theta = [1; 1.5];
tests(2).cb = 1;
tests(2).Nsamp = 1024;

tests(3).name = 'three-user mimo';
tests(3).H = (randn(3, 6, 2) + 1j * randn(3, 6, 2)) / sqrt(2);
tests(3).Lxu = [2 2 2];
tests(3).Etotal = 9;
tests(3).theta = [1.2; 0.8; 1.6];
tests(3).cb = 1;
tests(3).Nsamp = 1024;

tests(4).name = 'stress mimo';
tests(4).H = (randn(4, 9, 8) + 1j * randn(4, 9, 8)) / sqrt(2);
tests(4).Lxu = [2 1 2 1 2 1];
tests(4).Etotal = 18;
tests(4).theta = [1; 2; 1.5; 0.5; 1.2; 0.8];
tests(4).cb = 1;
tests(4).Nsamp = 1024;

fprintf('[adaptive maxresmac mimo]\n\n');

results = [];

for t = 1:numel(tests)
    result = run_adaptive_test(tests(t), t);
    results = [results; result];
    fprintf('\n');
end

fprintf('[summary]\n');
fprintf('%-22s | %10s %10s | %10s %10s | %8s %8s | %9s %9s\n', ...
    'test', 'rate[canon]', 'rate[ada]', 'energy[c]', 'energy[a]', 'ms[c]', 'ms[a]', 'rel gap', 'prac gap');
fprintf('%s\n', repmat('-', 1, 114));

all_passed = true;

for t = 1:numel(results)
    r = results(t);
    all_passed = all_passed && r.passed;
    fprintf('%-22s | %10.5f %10.5f | %10.5f %10.5f | %8.2f %8.2f | %9.2e %9.2e\n', ...
        r.name, r.canonical_weighted_rate, r.adaptive_weighted_rate, ...
        r.canonical_energy, r.adaptive_energy, r.canonical_elapsed_ms, r.adaptive_elapsed_ms, ...
        r.relative_gap, r.practical_gap);
end

% [subband grouping] solve on a coarser frequency grid (one design per subband,
% applied to all its tones) to cut feedback and the tones that must be probed
fprintf('\n[subband grouping]\n');
fprintf('%-12s | %11s %11s %10s | %8s\n', 'grid', 'rate[canon]', 'rate[ada]', 'rel gap', 'designs');
fprintf('%s\n', repmat('-', 1, 62));

sb_tc = tests(4);
sb_theta = sb_tc.theta(:);
N_full = size(sb_tc.H, 3);

for Nsb = [N_full, 4, 2]
    Hred = group_subbands_ref(sb_tc.H, Nsb);
    [~, ~, ~, bc_sb] = maxRESMACMIMO(Hred, sb_tc.Lxu, sb_tc.Etotal, sb_tc.theta, sb_tc.cb, false);

    opts_sb = struct('Nsamp', sb_tc.Nsamp, 'maxOuter', 60, 'rngSeed', 2000, 'verbose', false);

    if Nsb < N_full
        opts_sb.subbands = Nsb;
    end

    [Rada_sb, ~, ~, ~] = ada_maxRESMACMIMO(sb_tc.H, sb_tc.Lxu, sb_tc.Etotal, sb_tc.theta, sb_tc.cb, opts_sb);

    rc_sb = sb_theta.' * sum(bc_sb, 2);
    ra_sb = true_weighted_rate(Hred, Rada_sb, sb_tc.Lxu, sb_theta, sb_tc.cb);
    gap_sb = abs(ra_sb - rc_sb) / max(1, abs(rc_sb));
    n_designs = numel(sb_tc.Lxu) * Nsb;

    if Nsb == N_full
        label = sprintf('%d (full)', Nsb);
    else
        label = sprintf('%d subbands', Nsb);
    end

    fprintf('%-12s | %11.5f %11.5f %10.2e | %8d\n', label, rc_sb, ra_sb, gap_sb, n_designs);
end

if ~all_passed
    error('one or more adaptive maxRESMAC MIMO tests failed');
end

function Hsb = group_subbands_ref(H, Nsb)
    % mirror of the solver's internal grouping so the canonical comparison sees
    % the same representative tones
    N = size(H, 3);
    edges = round(linspace(0, N, Nsb + 1));
    Hsb = zeros(size(H, 1), size(H, 2), Nsb);

    for s = 1:Nsb
        idx = (edges(s) + 1):edges(s + 1);
        Hsb(:, :, s) = H(:, :, idx(ceil(numel(idx) / 2)));
    end

end

function [wr, b] = true_weighted_rate(H, Rxx, Lxu, theta, cb)
    % weighted sum rate a covariance design achieves on the true channel, using
    % the SIC order (descending theta) and whitened cumulative log-dets. Uses H
    % for validation only, not in the learner
    [Ly, ~, N] = size(H);
    theta = theta(:);
    U = numel(theta);

    if isscalar(Lxu)
        Lxu = ones(1, U) * Lxu;
    end

    idx_end = cumsum(Lxu);
    idx_start = [1, idx_end(1:end - 1) + 1];
    [~, order] = sort(theta, 'descend');
    is_cell = iscell(Rxx);
    b = zeros(U, N);

    for n = 1:N
        Sprev = eye(Ly);

        for k = 1:U
            u = order(k);
            cols = idx_start(u):idx_end(u);
            Hu = H(:, cols, n);

            if is_cell
                Ru = Rxx{u, n};
            else
                Ru = Rxx(1:Lxu(u), 1:Lxu(u), u, n);
            end

            HRH = 0.5 * (Hu * Ru * Hu' + (Hu * Ru * Hu')');
            Scur = 0.5 * (Sprev + HRH + (Sprev + HRH)');
            b(u, n) = (logdet_psd(Scur) - logdet_psd(Sprev)) / (log(2) * cb);
            Sprev = Scur;
        end

    end

    wr = theta.' * sum(b, 2);
end

function v = logdet_psd(M)
    M = 0.5 * (M + M');
    [L, p] = chol(M);

    if p == 0
        v = 2 * sum(log(diag(L)));
    else
        v = sum(log(max(real(eig(M)), 1e-15)));
    end

end

function result = run_adaptive_test(tc, seed_offset)
    fprintf('[test] %s\n', tc.name);

    opts = struct();
    opts.Nsamp = tc.Nsamp;
    opts.maxOuter = 60;
    opts.rngSeed = 1000 + seed_offset;
    opts.verbose = false;

    tic;
    [Rcanon, Ecanon, wcanon, bcanon] = maxRESMACMIMO(tc.H, tc.Lxu, tc.Etotal, tc.theta, tc.cb, false);
    tcanon = toc;

    tic;
    [Rada, Eada, wada, bada, info] = ada_maxRESMACMIMO(tc.H, tc.Lxu, tc.Etotal, tc.theta, tc.cb, opts);
    tada = toc;

    % [practical path] deployable blind surrogate optimizer: batched fresh-noise
    % probes of the rate oracle, a noise-averaging quadratic surrogate fit, and an
    % analytic trust-region step (no common random numbers, no channel formed)
    opts_prac = opts;
    opts_prac.mode = 'practical';
    [Rprac, ~, ~, ~, infoprac] = ada_maxRESMACMIMO(tc.H, tc.Lxu, tc.Etotal, tc.theta, tc.cb, opts_prac);

    theta = tc.theta(:);
    canonical_weighted_rate = theta.' * sum(bcanon, 2);

    [adaptive_weighted_rate, bada_true] = true_weighted_rate(tc.H, Rada, tc.Lxu, theta, tc.cb);
    practical_rate = true_weighted_rate(tc.H, Rprac, tc.Lxu, theta, tc.cb);
    oracle_weighted_rate = theta.' * sum(bada, 2);
    relative_gap = abs(adaptive_weighted_rate - canonical_weighted_rate) / max(1, abs(canonical_weighted_rate));
    practical_gap = abs(practical_rate - canonical_weighted_rate) / max(1, abs(canonical_weighted_rate));
    energy_err = abs(sum(Eada, 'all') - tc.Etotal) / max(tc.Etotal, 1e-12);
    per_user_err = norm(sum(bada_true, 2) - sum(bcanon, 2)) / max(1, norm(sum(bcanon, 2)));
    spectrum_err = covariance_spectrum_error(Rada, Rcanon);
    w_err = norm(wada - wcanon) / max(1, norm(wcanon));

    result = struct();
    result.name = tc.name;
    result.canonical_weighted_rate = canonical_weighted_rate;
    result.adaptive_weighted_rate = adaptive_weighted_rate;
    result.canonical_energy = sum(Ecanon, 'all');
    result.adaptive_energy = sum(Eada, 'all');
    result.canonical_elapsed_ms = tcanon * 1000;
    result.adaptive_elapsed_ms = tada * 1000;
    result.relative_gap = relative_gap;
    result.practical_gap = practical_gap;
    result.practical_iters = infoprac.iters;
    result.practical_probes = infoprac.probes;
    result.practical_rounds = infoprac.probeRounds;
    result.energy_err = energy_err;
    result.per_user_err = per_user_err;
    result.spectrum_err = spectrum_err;
    result.w_err = w_err;
    result.passed = true;

    fprintf('  canonical per-user rates: [%s]\n', num2str(sum(bcanon, 2)'));
    fprintf('  adaptive per-user rates:  [%s]\n', num2str(sum(bada_true, 2)'));
    fprintf('  canonical weighted rate: %.8f\n', canonical_weighted_rate);
    fprintf('  adaptive weighted rate:  %.8f  (oracle-measured %.6f)\n', adaptive_weighted_rate, oracle_weighted_rate);
    fprintf('  relative weighted gap:   %.3e\n', relative_gap);
    fprintf('  practical-mode gap:      %.3e  (rate %.6f)\n', practical_gap, practical_rate);
    fprintf('  practical probes/rounds: %d / %d (%d pilots each, fresh noise)\n', ...
        infoprac.probes, infoprac.probeRounds, infoprac.probePilots);
    fprintf('  per-user rate error:     %.3e\n', per_user_err);
    fprintf('  covariance spectrum err: %.3e\n', spectrum_err);
    fprintf('  w relative error:        %.3e\n', w_err);
    fprintf('  adaptive energy used:    %.8f\n', sum(Eada, 'all'));
    fprintf('  canonical/adaptive ms:   %.2f / %.2f\n', result.canonical_elapsed_ms, result.adaptive_elapsed_ms);
    fprintf('  adaptive mode/iters:     %s / %d\n', info.mode, info.iters);

    if relative_gap > 1e-2
        fprintf('  [x] weighted rate mismatch\n');
        result.passed = false;
    end

    if practical_gap > 5e-2
        fprintf('  [x] practical (blind) gap too large\n');
        result.passed = false;
    end

    if per_user_err > 1.1e-1
        fprintf('  [x] per-user rate mismatch\n');
        result.passed = false;
    end

    if spectrum_err > 1e-1
        fprintf('  [x] covariance spectra mismatch\n');
        result.passed = false;
    end

    if energy_err > 1e-8
        fprintf('  [x] adaptive energy constraint mismatch\n');
        result.passed = false;
    end

    if ~check_psd(Rada)
        fprintf('  [x] adaptive covariance is not PSD\n');
        result.passed = false;
    end

    if result.passed
        fprintf('  [p] test passed\n');
    end

end

function err = covariance_spectrum_error(Ra, Rc)
    vals_a = [];
    vals_c = [];

    if iscell(Ra)

        for u = 1:size(Ra, 1)

            for n = 1:size(Ra, 2)
                vals_a = [vals_a; sort(real(eig(0.5 * (Ra{u, n} + Ra{u, n}'))), 'descend')];
                vals_c = [vals_c; sort(real(eig(0.5 * (Rc{u, n} + Rc{u, n}'))), 'descend')];
            end

        end

    else

        for u = 1:size(Ra, 3)

            for n = 1:size(Ra, 4)
                A = Ra(:, :, u, n);
                C = Rc(:, :, u, n);
                vals_a = [vals_a; sort(real(eig(0.5 * (A + A'))), 'descend')];
                vals_c = [vals_c; sort(real(eig(0.5 * (C + C'))), 'descend')];
            end

        end

    end

    err = norm(vals_a - vals_c) / max(1, norm(vals_c));
end

function ok = check_psd(Rxxs)
    ok = true;

    if iscell(Rxxs)

        for u = 1:size(Rxxs, 1)

            for n = 1:size(Rxxs, 2)
                R = Rxxs{u, n};

                if any(real(eig(0.5 * (R + R'))) < -1e-8)
                    ok = false;
                    return;
                end

            end

        end

    else

        for u = 1:size(Rxxs, 3)

            for n = 1:size(Rxxs, 4)
                R = Rxxs(:, :, u, n);

                if any(real(eig(0.5 * (R + R'))) < -1e-8)
                    ok = false;
                    return;
                end

            end

        end

    end

end
