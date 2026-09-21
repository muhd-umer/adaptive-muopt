%TEST_ADA_MAXRESMACMIMO Compare adaptive and canonical maxRESMAC solvers

clear;
clc;
rng(42);

require_mex('maxresmac_mex');

tests = struct();

tests(1).name = 'siso';
tests(1).H = reshape([4 1], 1, 2, 1);
tests(1).Lxu = [1 1];
tests(1).Etotal = 2;
tests(1).theta = [1; 1];
tests(1).cb = 1;
tests(1).Rnn = 1;
tests(1).maxOuter = 100;
tests(1).gapTol = 2e-5;

H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(2).name = 'basic mimo';
tests(2).H = H;
tests(2).Lxu = [2 2];
tests(2).Etotal = 10;
tests(2).theta = [1; 1];
tests(2).cb = 1;
tests(2).Rnn = eye(2);
tests(2).maxOuter = 120;
tests(2).gapTol = 2e-3;

tests(3).name = 'unequal antennas';
tests(3).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(3).Lxu = [2 3];
tests(3).Etotal = 8;
tests(3).theta = [1; 1.5];
tests(3).cb = 1;
tests(3).Rnn = eye(3);
tests(3).maxOuter = 120;
tests(3).gapTol = 5e-3;

H = zeros(3, 4, 2);
H(:, :, 1) = [1.2 0.4 0.7 0.2; 0.3 1.1 0.4 0.8; 0.9 0.2 0.6 0.1];
H(:, :, 2) = [0.7 0.5 1.0 0.3; 0.2 0.9 0.5 0.7; 0.5 0.3 0.8 0.2];
tests(4).name = 'colored rank deficient';
tests(4).H = H;
tests(4).Lxu = [2 2];
tests(4).Etotal = 5;
tests(4).theta = [1.4; 0.7];
tests(4).cb = 1;
tests(4).Rnn = [1.0 0.2 0.1; 0.2 1.4 0.15; 0.1 0.15 0.8];
tests(4).maxOuter = 120;
tests(4).gapTol = 5e-3;

H = zeros(2, 2, 2);
H(:, :, 1) = [1.4 0.3 + 0.2j; 0.1 - 0.1j 0.8];
H(:, :, 2) = [0.9 - 0.2j 0.4; 0.2 1.2 + 0.1j];
tests(5).name = 'single user mimo';
tests(5).H = H;
tests(5).Lxu = 2;
tests(5).Etotal = 4;
tests(5).theta = 1.2;
tests(5).cb = 1;
tests(5).Rnn = eye(2);
tests(5).maxOuter = 100;
tests(5).gapTol = 2e-3;

tests(6).name = 'stress';
tests(6).H = (randn(4, 9, 64) + 1j * randn(4, 9, 64)) / sqrt(2);
tests(6).Lxu = [2 1 2 1 2 1];
tests(6).Etotal = 30;
tests(6).theta = [1; 2; 1.5; 0.5; 1.2; 0.8];
tests(6).cb = 1;
tests(6).Rnn = eye(4);
tests(6).maxOuter = 100;
tests(6).gapTol = 1e-2;

fprintf('[adaptive maxresmac mimo]\n\n');
results = repmat(empty_result(), numel(tests), 1);

for t = 1:numel(tests)
    results(t) = run_test(tests(t), t);
    fprintf('\n');
end

singlePassPassed = run_single_pass_test();
zeroEnergyPassed = run_zero_energy_test();
contractPassed = run_contract_test();
fprintf('\n');

fprintf('[summary]\n');
fprintf('%-24s | %11s %11s | %10s | %10s %10s\n', ...
    'test', 'rate[canon]', 'rate[ada]', 'rel gap', 'ms[canon]', 'ms[ada]');
fprintf('%s\n', repmat('-', 1, 90));

for t = 1:numel(results)
    result = results(t);
    fprintf('%-24s | %11.5f %11.5f | %10.2e | %10.2f %10.2f\n', ...
        result.name, result.canonicalObjective, result.adaptiveObjective, ...
        result.relativeGap, result.canonicalMs, result.adaptiveMs);
end

if ~all([results.passed]) || ~singlePassPassed || ...
        ~zeroEnergyPassed || ~contractPassed
    error('one or more adaptive maxRESMACMIMO tests failed');
end

function passed = run_single_pass_test()
    calls = 0;
    H = [1.5 0.2 + 0.1j; 0.1 - 0.2j 0.7];
    pilot = struct('numRx', 2, 'numTones', 1, ...
        'sample', @sample);
    opts = struct('R0', eye(2), 'numPilots', 32, 'singlePass', true, ...
        'stepInitial', 0.05, 'floorInitial', 1e-6, 'rngSeed', 91);
    initial = oracle_weighted_rate(H, eye(2), 2, 1, 1);
    [R, E, ~, b, info] = ada_maxRESMACMIMO(pilot, 2, 2, 1, 1, opts);
    updated = oracle_weighted_rate(H, R, 2, 1, 1);
    observedError = norm(info.observedR{1} - eye(2), 'fro');
    passed = calls == 1 && info.pilotCalls == 1 && ...
        info.iterations == 1 && abs(sum(E, 'all') - 2) <= 1e-10 && ...
        check_psd(R) && updated >= initial - 1e-10 && ...
        observedError <= 1e-10 && all(isfinite(b(:)));
    fprintf('[test] causal single pass: calls=%d, rate %.6f -> %.6f [%s]\n', ...
        calls, initial, updated, pass_label(passed));

    function Y = sample(~, Atx, X)
        calls = calls + 1;
        Y = exact_pilot_samples(H, eye(size(H, 1)), Atx, X);
    end
end

function passed = run_zero_energy_test()
    pilot = struct('numRx', 2, 'numTones', 1, ...
        'sample', @(n, Atx, X) zero_energy_samples(n, Atx, X));
    opts = struct('numPilots', 16, 'singlePass', true, 'rngSeed', 92);
    [R, E, w, b, info] = ada_maxRESMACMIMO(pilot, 2, 0, 1, 1, opts);
    passed = sum(E, 'all') == 0 && all(R(:) == 0) && w == 0 && ...
        all(b(:) == 0) && info.objective == 0 && info.pilotCalls == 1;
    fprintf('[test] zero-energy single pass [%s]\n', pass_label(passed));
end

function passed = run_contract_test()
    source = fileread(which('ada_maxRESMACMIMO'));
    forbidden = {'\<H\>', '\<Rnn\>', '\<mu_mac\>', '\<maxRESMACMIMO\s*\('};
    passed = contains(source, 'pilot.sample') && ...
        all(cellfun(@(pattern) isempty(regexp(source, pattern, 'once')), forbidden));
    fprintf('[test] non-cheating source contract [%s]\n', pass_label(passed));
end

function Y = zero_energy_samples(~, ~, X)
    Y = zeros(2, size(X, 2));
end

function label = pass_label(passed)
    if passed
        label = 'p';
    else
        label = 'x';
    end
end

function result = run_test(tc, seedOffset)
    fprintf('[test] %s\n', tc.name);
    Lnoise = chol(tc.Rnn, 'lower');
    Hwhite = tc.H;

    for n = 1:size(tc.H, 3)
        Hwhite(:, :, n) = Lnoise \ tc.H(:, :, n);
    end

    pilot = struct();
    pilot.numRx = size(tc.H, 1);
    pilot.numTones = size(tc.H, 3);
    pilot.sample = @(n, Atx, X) exact_pilot_samples( ...
        tc.H(:, :, n), Lnoise, Atx, X);
    opts = struct();
    opts.maxOuter = tc.maxOuter;
    opts.numPilots = max(64, 8 * (sum(tc.Lxu) + pilot.numRx));
    opts.rngSeed = 700 + seedOffset;
    opts.tolObjective = 1e-8;
    opts.tolCovariance = 1e-7;
    opts.patience = 4;

    tic;
    [~, Ecanonical, wcanonical, bcanonical] = maxRESMACMIMO( ...
        Hwhite, tc.Lxu, tc.Etotal, tc.theta, tc.cb, true);
    canonicalMs = toc * 1000;
    tic;
    [Radaptive, Eadaptive, wadaptive, badaptive, info] = ada_maxRESMACMIMO( ...
        pilot, tc.Lxu, tc.Etotal, tc.theta, tc.cb, opts);
    adaptiveMs = toc * 1000;

    canonicalObjective = tc.theta(:)' * sum(bcanonical, 2);
    adaptiveObjective = oracle_weighted_rate( ...
        Hwhite, Radaptive, tc.Lxu, tc.theta, tc.cb);
    learnedObjective = tc.theta(:)' * sum(badaptive, 2);
    relativeGap = abs(adaptiveObjective - canonicalObjective) / ...
        max(1, abs(canonicalObjective));
    learnedGap = abs(learnedObjective - adaptiveObjective) / ...
        max(1, abs(adaptiveObjective));
    energyError = abs(sum(Eadaptive, 'all') - tc.Etotal) / max(tc.Etotal, 1);
    closureError = learned_closure_error(info);
    priorityGap = max(abs(sum(badaptive, 2) - sum(bcanonical, 2))) / ...
        max(1, max(sum(bcanonical, 2)));
    orderMatch = isequal(info.order(:), priority_order(tc.theta));
    [dualResidual, complementarityResidual] = oracle_kkt_residual( ...
        Hwhite, Radaptive, tc.Lxu, tc.theta, tc.cb);
    passed = relativeGap <= tc.gapTol && learnedGap <= 1e-8 && ...
        energyError <= 1e-10 && closureError <= 1e-10 && ...
        priorityGap <= tc.gapTol && orderMatch && ...
        dualResidual <= 2e-2 && complementarityResidual <= 2e-2 && ...
        check_psd(Radaptive);

    fprintf('  canonical/adaptive objective: %.8f / %.8f\n', ...
        canonicalObjective, adaptiveObjective);
    fprintf('  relative objective gap:       %.3e\n', relativeGap);
    fprintf('  learned-rate gap:             %.3e\n', learnedGap);
    fprintf('  per-user priority-rate gap:   %.3e (order match %d)\n', ...
        priorityGap, orderMatch);
    fprintf('  canonical/adaptive energy:    %.8f / %.8f\n', ...
        sum(Ecanonical, 'all'), sum(Eadaptive, 'all'));
    fprintf('  canonical/adaptive w:         %.6g / %.6g\n', ...
        wcanonical(1), wadaptive(1));
    fprintf('  G/S0 closure error:           %.3e\n', closureError);
    fprintf('  KKT dual/complementarity:     %.3e / %.3e\n', ...
        dualResidual, complementarityResidual);
    fprintf('  canonical/adaptive time:      %.2f / %.2f ms\n', ...
        canonicalMs, adaptiveMs);
    fprintf('  iterations/pilot calls:       %d / %d\n', ...
        info.iterations, info.pilotCalls);

    if passed
        fprintf('  [p] test passed\n');
    else
        fprintf('  [x] test failed\n');
    end

    result = empty_result();
    result.name = tc.name;
    result.canonicalObjective = canonicalObjective;
    result.adaptiveObjective = adaptiveObjective;
    result.relativeGap = relativeGap;
    result.canonicalMs = canonicalMs;
    result.adaptiveMs = adaptiveMs;
    result.passed = passed;
end

function result = empty_result()
    result = struct('name', '', 'canonicalObjective', 0, ...
        'adaptiveObjective', 0, 'relativeGap', inf, ...
        'canonicalMs', 0, 'adaptiveMs', 0, 'passed', false);
end

function order = priority_order(theta)
    [~, order] = sort(theta(:), 'descend');
    order = order(:);
end

function Y = exact_pilot_samples(channel, noiseFactor, Atx, X)
    count = size(X, 2);
    Ly = size(channel, 1);
    indices = 0:count - 1;
    seed = (1:Ly)';
    Z = exp(2j * pi * seed * indices / count);
    Z = Z - (Z * X') * X / count;
    covariance = (Z * Z') / count;
    covariance = 0.5 * (covariance + covariance');
    factor = chol(covariance, 'lower');
    Z = factor \ Z;
    Y = channel * Atx * X + noiseFactor * Z;
end

function objective = oracle_weighted_rate(H, Rxx, Lxu, theta, cb)
    theta = theta(:);
    U = numel(theta);
    N = size(H, 3);
    ranges = user_ranges(Lxu, U);
    [~, order] = sort(theta, 'descend');
    R = covariance_cells(Rxx, Lxu, U, N);
    rates = zeros(U, N);

    for n = 1:N
        Sprevious = eye(size(H, 1));

        for k = 1:U
            u = order(k);
            Hu = H(:, ranges{u}, n);
            Scurrent = Sprevious + Hu * R{u, n} * Hu';
            rates(u, n) = (logdet_spd(Scurrent) - logdet_spd(Sprevious)) / ...
                (cb * log(2));
            Sprevious = Scurrent;
        end
    end

    objective = theta' * sum(rates, 2);
end

function error = learned_closure_error(info)
    numerator = 0;
    denominator = 0;

    for n = 1:numel(info.J)
        reconstructed = info.G{n}' * diag(info.S0{n}) * info.G{n};
        numerator = numerator + norm(info.J{n} - reconstructed, 'fro') ^ 2;
        denominator = denominator + norm(info.J{n}, 'fro') ^ 2;
    end

    error = sqrt(numerator / max(denominator, realmin));
end

function [dualResidual, complementarityResidual] = oracle_kkt_residual( ...
        H, Rxx, Lxu, theta, cb)
    theta = theta(:);
    U = numel(theta);
    N = size(H, 3);
    ranges = user_ranges(Lxu, U);
    R = covariance_cells(Rxx, Lxu, U, N);
    [stheta, order] = sort(theta, 'descend');
    delta = stheta - [stheta(2:end); 0];
    gradient = cell(U, N);

    for n = 1:N
        inverses = cell(U, 1);
        S = eye(size(H, 1));

        for k = 1:U
            u = order(k);
            Hu = H(:, ranges{u}, n);
            S = S + Hu * R{u, n} * Hu';
            inverses{k} = S \ eye(size(S));
        end

        tail = zeros(size(S));

        for k = U:-1:1
            tail = tail + delta(k) * inverses{k};
            u = order(k);
            Hu = H(:, ranges{u}, n);
            gradient{u, n} = Hu' * tail * Hu / (cb * log(2));
            gradient{u, n} = 0.5 * (gradient{u, n} + gradient{u, n}');
        end
    end

    marginalParts = cell(numel(R), 1);

    for i = 1:numel(R)
        [V, D] = eig(0.5 * (R{i} + R{i}'));
        active = real(diag(D)) > 1e-7 * max(1, real(trace(R{i})));

        if any(active)
            marginalParts{i} = real(diag( ...
                V(:, active)' * gradient{i} * V(:, active)));
        end
    end

    activeMarginals = vertcat(marginalParts{:});
    lambda = median(activeMarginals);
    dualResidual = 0;
    complementarityResidual = 0;

    for i = 1:numel(R)
        slack = lambda * eye(size(R{i})) - gradient{i};
        slack = 0.5 * (slack + slack');
        dualResidual = max(dualResidual, max(0, -min(real(eig(slack)))));
        complementarityResidual = max( ...
            complementarityResidual, norm(slack * R{i}, 'fro'));
    end

    dualResidual = dualResidual / max(1, abs(lambda));
    complementarityResidual = complementarityResidual / max(1, abs(lambda));
end

function R = covariance_cells(Rxx, Lxu, U, N)
    if iscell(Rxx)
        R = Rxx;
        return;
    end

    if isscalar(Lxu)
        Lxu = repmat(Lxu, 1, U);
    end

    R = cell(U, N);

    for u = 1:U
        for n = 1:N
            R{u, n} = Rxx(1:Lxu(u), 1:Lxu(u), u, n);
        end
    end
end

function ranges = user_ranges(Lxu, U)
    if isscalar(Lxu)
        Lxu = repmat(Lxu, 1, U);
    end

    ends = cumsum(Lxu);
    starts = [1, ends(1:end - 1) + 1];
    ranges = cell(1, U);

    for u = 1:U
        ranges{u} = starts(u):ends(u);
    end
end

function value = logdet_spd(matrix)
    factor = chol(0.5 * (matrix + matrix'));
    value = 2 * sum(log(real(diag(factor))));
end

function ok = check_psd(Rxxs)
    ok = true;

    if iscell(Rxxs)
        blocks = Rxxs;
    else
        blocks = cell(size(Rxxs, 3), size(Rxxs, 4));

        for u = 1:size(Rxxs, 3)
            for n = 1:size(Rxxs, 4)
                blocks{u, n} = Rxxs(:, :, u, n);
            end
        end
    end

    for i = 1:numel(blocks)
        covariance = 0.5 * (blocks{i} + blocks{i}');

        if any(real(eig(covariance)) < -1e-9)
            ok = false;
            return;
        end
    end
end

function require_mex(name)

    if exist(name, 'file') ~= 3
        error('test:mexMissing', ...
            'canonical reference %s not found; compile it before benchmarking', name);
    end

end
