%TEST_ADA_MAXRMACMIMO Compare adaptive and canonical maxRMAC solvers

clear;
clc;
rng(42);

require_mex('maxrmac_mex');

tests = struct();

tests(1).name = 'siso';
tests(1).H = reshape([4 1], 1, 2, 1);
tests(1).Lxu = [1 1];
tests(1).Eu = [2; 2];
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
tests(2).Eu = [5; 6];
tests(2).theta = [1; 1];
tests(2).cb = 1;
tests(2).Rnn = eye(2);
tests(2).maxOuter = 120;
tests(2).gapTol = 2e-3;

tests(3).name = 'unequal weights';
tests(3).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(3).Lxu = [2 3];
tests(3).Eu = [4; 6];
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
tests(4).Eu = [3; 2];
tests(4).theta = [1.4; 0.7];
tests(4).cb = 1;
tests(4).Rnn = [1.0 0.2 0.1; 0.2 1.4 0.15; 0.1 0.15 0.8];
tests(4).maxOuter = 120;
tests(4).gapTol = 5e-3;

tests(5).name = 'complex mimo';
tests(5).H = (randn(2, 4, 3) + 1j * randn(2, 4, 3)) / sqrt(2);
tests(5).Lxu = [2 2];
tests(5).Eu = [3; 5];
tests(5).theta = [0.8; 1.3];
tests(5).cb = 1;
tests(5).Rnn = eye(2);
tests(5).maxOuter = 120;
tests(5).gapTol = 5e-3;

tests(6).name = 'stress';
tests(6).H = (randn(4, 9, 64) + 1j * randn(4, 9, 64)) / sqrt(2);
tests(6).Lxu = [2 1 2 1 2 1];
tests(6).Eu = [5; 4; 6; 3; 5; 4];
tests(6).theta = [1; 2; 1.5; 0.5; 1.2; 0.8];
tests(6).cb = 1;
tests(6).Rnn = eye(4);
tests(6).maxOuter = 100;
tests(6).gapTol = 1e-2;

fprintf('[adaptive maxrmac mimo]\n\n');
results = repmat(empty_result(), numel(tests), 1);

for t = 1:numel(tests)
    results(t) = run_test(tests(t), t);
    fprintf('\n');
end

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

if ~all([results.passed]) || ~contractPassed
    error('one or more adaptive maxRMACMIMO tests failed');
end

function passed = run_contract_test()
    source = fileread(which('ada_maxRMACMIMO'));
    forbidden = {'\<H\>', '\<Rnn\>', '\<mu_mac\>', '\<maxRMACMIMO\s*\('};
    passed = contains(source, 'pilot.sample') && ...
        all(cellfun(@(pattern) isempty(regexp(source, pattern, 'once')), forbidden));
    fprintf('[test] non-cheating source contract [%s]\n', pass_label(passed));
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
    [~, Ecanonical, wcanonical, bcanonical] = maxRMACMIMO( ...
        Hwhite, tc.Lxu, tc.Eu, tc.theta, tc.cb, true);
    canonicalMs = toc * 1000;
    tic;
    [Radaptive, Eadaptive, wadaptive, badaptive, info] = ada_maxRMACMIMO( ...
        pilot, tc.Lxu, tc.Eu, tc.theta, tc.cb, opts);
    adaptiveMs = toc * 1000;

    canonicalObjective = tc.theta(:)' * sum(bcanonical, 2);
    adaptiveObjective = oracle_weighted_rate( ...
        Hwhite, Radaptive, tc.Lxu, tc.theta, tc.cb);
    learnedObjective = tc.theta(:)' * sum(badaptive, 2);
    relativeGap = abs(adaptiveObjective - canonicalObjective) / ...
        max(1, abs(canonicalObjective));
    learnedGap = abs(learnedObjective - adaptiveObjective) / ...
        max(1, abs(adaptiveObjective));
    energyPerUser = sum(Eadaptive, 2);
    energyError = max(abs(energyPerUser - tc.Eu(:)) ./ max(tc.Eu(:), 1e-12));
    closureError = learned_closure_error(info);
    [dualResidual, complementarityResidual] = oracle_kkt_residual( ...
        Hwhite, Radaptive, tc.Lxu, tc.theta, tc.cb);
    passed = relativeGap <= tc.gapTol && learnedGap <= 1e-8 && ...
        energyError <= 1e-8 && closureError <= 1e-10 && ...
        dualResidual <= 2e-2 && complementarityResidual <= 2e-2 && ...
        check_psd(Radaptive);

    fprintf('  canonical/adaptive objective: %.8f / %.8f\n', ...
        canonicalObjective, adaptiveObjective);
    fprintf('  relative objective gap:       %.3e\n', relativeGap);
    fprintf('  learned-rate gap:             %.3e\n', learnedGap);
    fprintf('  per-user energy error:        %.3e\n', energyError);
    fprintf('  canonical/adaptive w(1):      %.6g / %.6g\n', ...
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

    dualResidual = 0;
    complementarityResidual = 0;

    for u = 1:U
        marginalParts = cell(N, 1);

        for n = 1:N
            [V, D] = eig(0.5 * (R{u, n} + R{u, n}'));
            active = real(diag(D)) > 1e-7 * max(1, real(trace(R{u, n})));

            if any(active)
                marginalParts{n} = real(diag( ...
                    V(:, active)' * gradient{u, n} * V(:, active)));
            end
        end

        activeMarginals = vertcat(marginalParts{:});

        if isempty(activeMarginals)
            continue;
        end

        lambda = median(activeMarginals);

        for n = 1:N
            slack = lambda * eye(size(R{u, n})) - gradient{u, n};
            slack = 0.5 * (slack + slack');
            dualResidual = max(dualResidual, ...
                max(0, -min(real(eig(slack)))) / max(1, abs(lambda)));
            complementarityResidual = max(complementarityResidual, ...
                norm(slack * R{u, n}, 'fro') / max(1, abs(lambda)));
        end
    end
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
