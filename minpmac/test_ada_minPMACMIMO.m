%TEST_ADA_MINPMACMIMO Compare adaptive and canonical minPMAC solvers

clear;
clc;
rng(42);

require_mex('minpmac_mex');

tests = struct();

Href = zeros(1, 2, 4);
Href(:, :, 1) = [80 60];
Href(:, :, 2) = [40 30];
Href(:, :, 3) = [50 50];
Href(:, :, 4) = [30 40];

tests(1).name = 'reference siso';
tests(1).H = Href;
tests(1).Lxu = [1 1];
tests(1).bu_min = [24 24];
tests(1).w = [1 1];
tests(1).cb = 1;
tests(1).Rnn = 1;
tests(1).gapTol = 5e-3;
tests(1).certTol = 1e-6;

H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(2).name = 'reference mimo';
tests(2).H = H;
tests(2).Lxu = [2 2];
tests(2).bu_min = [10 10];
tests(2).w = [1 1];
tests(2).cb = 1;
tests(2).Rnn = eye(2);
tests(2).gapTol = 5e-3;
tests(2).certTol = 1e-6;

tests(3).name = 'unequal weights';
tests(3).H = Href;
tests(3).Lxu = [1 1];
tests(3).bu_min = [20 20];
tests(3).w = [1 2];
tests(3).cb = 1;
tests(3).Rnn = 1;
tests(3).gapTol = 5e-3;
tests(3).certTol = 1e-6;

tests(4).name = 'mimo unequal ant';
tests(4).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(4).Lxu = [2 3];
tests(4).bu_min = [4 6];
tests(4).w = [1 1.5];
tests(4).cb = 1;
tests(4).Rnn = eye(3);
tests(4).gapTol = 5e-3;
tests(4).certTol = 1e-6;

H = zeros(3, 4, 2);
H(:, :, 1) = [1.2 0.4 0.7 0.2; 0.3 1.1 0.4 0.8; 0.9 0.2 0.6 0.1];
H(:, :, 2) = [0.7 0.5 1.0 0.3; 0.2 0.9 0.5 0.7; 0.5 0.3 0.8 0.2];
tests(5).name = 'colored rank deficient';
tests(5).H = H;
tests(5).Lxu = [2 2];
tests(5).bu_min = [3 2];
tests(5).w = [1 1.4];
tests(5).cb = 1;
tests(5).Rnn = [1.0 0.2 0.1; 0.2 1.4 0.15; 0.1 0.15 0.8];
tests(5).gapTol = 5e-3;
tests(5).certTol = 1e-6;

H = zeros(1, 2, 4);
H(:, :, 1) = [5 4];
H(:, :, 2) = [3 2];
H(:, :, 3) = [2 3];
H(:, :, 4) = [4 5];
tests(6).name = 'real baseband';
tests(6).H = H;
tests(6).Lxu = [1 1];
tests(6).bu_min = [2 2];
tests(6).w = [1 1];
tests(6).cb = 2;
tests(6).Rnn = 1;
tests(6).gapTol = 5e-3;
tests(6).certTol = 1e-6;

tests(7).name = 'uniform Lxu';
tests(7).H = (randn(2, 6, 3) + 1j * randn(2, 6, 3)) / sqrt(2);
tests(7).Lxu = 2;
tests(7).bu_min = [3 4 5];
tests(7).w = [1 1 1];
tests(7).cb = 1;
tests(7).Rnn = eye(2);
tests(7).gapTol = 1e-2;
tests(7).certTol = 1e-6;

H = (randn(4, 9, 8) + 1j * randn(4, 9, 8)) / sqrt(2);
tests(8).name = 'tied users';
H(:, 3, :) = H(:, 1, :);
H(:, 4, :) = H(:, 2, :);
tests(8).H = H;
tests(8).Lxu = [2 2 2 2 1];
tests(8).bu_min = [3 3 3 3 2];
tests(8).w = [1 1 1 1 1];
tests(8).cb = 1;
tests(8).Rnn = eye(4);
tests(8).gapTol = 1e-2;
tests(8).certTol = 1e-6;

tests(9).name = 'stress';
tests(9).H = (randn(4, 9, 16) + 1j * randn(4, 9, 16)) / sqrt(2);
tests(9).Lxu = [2 1 2 1 2 1];
tests(9).bu_min = [2 1.8 2.2 1.5 2 1.8];
tests(9).w = [1 2 1.5 0.5 1.2 0.8];
tests(9).cb = 1;
tests(9).Rnn = eye(4);
tests(9).gapTol = 2e-2;
tests(9).certTol = 1e-6;

fprintf('[adaptive minpmac mimo]\n\n');
results = repmat(empty_result(), numel(tests), 1);

for t = 1:numel(tests)
    results(t) = run_test(tests(t), t);
    fprintf('\n');
end

trackingPassed = run_tracking_test();
contractPassed = run_contract_test();
fprintf('\n');

fprintf('[summary]\n');
fprintf('%-24s | %12s %12s | %11s | %11s | %5s %5s | %10s %10s\n', ...
    'test', 'pow[canon]', 'pow[ada]', 'excess', 'dual gap', 'f[c]', 'f[a]', ...
    'ms[canon]', 'ms[ada]');
fprintf('%s\n', repmat('-', 1, 118));

for t = 1:numel(results)
    result = results(t);
    fprintf('%-24s | %12.5f %12.5f | %+11.2e | %+11.2e | %5d %5d | %10.2f %10.2f\n', ...
        result.name, result.canonicalPower, result.adaptivePower, ...
        result.excess, result.dualGap, result.canonicalFlag, result.flag, ...
        result.canonicalMs, result.adaptiveMs);
end

if ~all([results.passed]) || ~contractPassed || ~trackingPassed
    error('one or more adaptive minPMACMIMO tests failed');
end

function passed = run_tracking_test()
    %RUN_TRACKING_TEST One measurement, one projected step, warm rate prices
    channel = reshape([3 1; 1 2], 2, 2, 1);
    Lxu = [1 1];
    calls = 0;
    pilot = struct();
    pilot.numRx = 2;
    pilot.numTones = 1;
    pilot.sample = @counted_sample;
    opts = struct('numPilots', 32, 'singlePass', true);
    opts.theta0 = [2; 2];
    opts.R0 = {0.5; 0.5};
    [flag, bu_a, info] = ada_minPMACMIMO(pilot, Lxu, [1 1], [1 1], 1, opts);
    passed = calls == 1 && info.pilotCalls == 1 && ...
        norm(info.observedR{1} - 0.5) <= 1e-10 && all(isfinite(bu_a)) && ...
        ismember(flag, [0, 1]);
    fprintf('[test] causal single pass: calls=%d, flag=%d [%s]\n', ...
        calls, flag, pass_label(passed));

    function Y = counted_sample(~, Atx, X)
        calls = calls + 1;
        Y = channel * Atx * X + 1e-3 * ones(2, size(X, 2));
    end

end

function passed = run_contract_test()
    source = fileread(which('ada_minPMACMIMO'));
    forbidden = {'\<H\>', '\<Rnn\>', '\<mu_mac\>', '\<minPMACMIMO\s*\('};
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
    opts.numPilots = max(64, 8 * (sum(expand_lxu(tc.Lxu, numel(tc.w))) + pilot.numRx));
    opts.rngSeed = 900 + seedOffset;

    tic;
    [canonicalFlag, ~, infoCanonical] = minPMACMIMO( ...
        Hwhite, tc.Lxu, tc.bu_min, tc.w, tc.cb, true);
    canonicalMs = toc * 1000;
    lastwarn('');
    tic;
    [flag, bu_a, info] = ada_minPMACMIMO( ...
        pilot, tc.Lxu, tc.bu_min, tc.w, tc.cb, opts);
    adaptiveMs = toc * 1000;
    [~, warnId] = lastwarn();

    weights = tc.w(:);
    canonicalPower = real(weights' * sum(infoCanonical.Eun, 2));
    adaptivePower = info.weightedPower;
    excess = (adaptivePower - canonicalPower) / max(1, abs(canonicalPower));
    relativeGap = abs(excess);
    rateSlack = min(bu_a(:) - tc.bu_min(:));
    learnedGap = learned_rate_gap(Hwhite, info, tc.Lxu, tc.cb);
    oracleGap = oracle_rate_gap(Hwhite, info, tc.Lxu, tc.cb, bu_a);
    closureError = learned_closure_error(info);
    priceRatio = info.theta(:) ./ max(infoCanonical.theta(:), realmin);
    psdOk = check_psd(info.Rxxs);

    dualGap = info.dualGap;
    certified = info.converged && abs(dualGap) <= tc.certTol && ...
        ~strcmp(warnId, 'ada_minPMACMIMO:priceNotConverged');

    passed = flag > 0 && rateSlack >= -1e-6 && excess <= tc.gapTol && ...
        learnedGap <= 1e-8 && oracleGap <= 1e-8 && ...
        closureError <= 1e-10 && psdOk && certified;

    fprintf('  canonical/adaptive flag:      %d / %d\n', canonicalFlag, flag);
    fprintf('  canonical/adaptive power:     %.8f / %.8f\n', ...
        canonicalPower, adaptivePower);
    fprintf('  relative power excess:        %+.3e\n', excess);
    fprintf('  target rates:                 [%s]\n', num2str(tc.bu_min, '%.4f '));
    fprintf('  achieved rates:               [%s]\n', num2str(bu_a, '%.4f '));
    fprintf('  worst rate slack:             %.3e\n', rateSlack);
    fprintf('  learned-rate gap:             %.3e\n', learnedGap);
    fprintf('  oracle rate-vector gap:       %.3e\n', oracleGap);
    fprintf('  G/S0 closure error:           %.3e\n', closureError);
    fprintf('  price ratio ada/canon:        [%s] (expected %.5f)\n', ...
        num2str(priceRatio', '%.5f '), tc.cb * log(2) / 2);
    fprintf('  restoration scale:            %.9f\n', info.restoreScale);
    fprintf('  certified duality gap:        %+.3e (%s)\n', ...
        dualGap, pass_label(certified));
    fprintf('  dual bounds lower/upper:      %.8f / %.8f\n', ...
        info.lowerBound, info.upperBound);
    fprintf('  canonical/adaptive time:      %.2f / %.2f ms\n', ...
        canonicalMs, adaptiveMs);
    fprintf('  price iterations/pilot calls: %d / %d\n', ...
        info.priceIterations, info.pilotCalls);

    if passed
        fprintf('  [p] test passed\n');
    else
        fprintf('  [x] test failed\n');
    end

    result = empty_result();
    result.name = tc.name;
    result.canonicalPower = canonicalPower;
    result.adaptivePower = adaptivePower;
    result.relativeGap = relativeGap;
    result.excess = excess;
    result.flag = flag;
    result.canonicalFlag = canonicalFlag;
    result.canonicalMs = canonicalMs;
    result.adaptiveMs = adaptiveMs;
    result.dualGap = dualGap;
    result.passed = passed;
end

function result = empty_result()
    result = struct('name', '', 'canonicalPower', 0, 'adaptivePower', 0, ...
        'relativeGap', inf, 'excess', inf, 'flag', 0, 'canonicalFlag', 0, ...
        'canonicalMs', 0, 'adaptiveMs', 0, 'dualGap', inf, 'passed', false);
end

function Lxu = expand_lxu(Lxu, U)

    if isscalar(Lxu)
        Lxu = repmat(Lxu, 1, U);
    else
        Lxu = reshape(Lxu, 1, []);
    end

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
    Z = chol(covariance, 'lower') \ Z;
    Y = channel * Atx * X + noiseFactor * Z;
end

function gap = learned_rate_gap(Hwhite, info, Lxu, cb)
    U = numel(info.theta);
    N = size(Hwhite, 3);
    Lxu = expand_lxu(Lxu, U);
    ranges = user_ranges(Lxu);
    R = covariance_cells(info.Rxxs, Lxu, U, N);
    oracle = oracle_order_rates(Hwhite, R, ranges, fliplr(info.order(:)'), cb);
    learned = sum(info.learnedRates, 2);
    gap = max(abs(learned - oracle)) / max(1, max(abs(oracle)));
end

function gap = oracle_rate_gap(Hwhite, info, Lxu, cb, bu_a)
    %ORACLE_RATE_GAP Recheck the reported rate vector against the true channel
    U = numel(info.theta);
    N = size(Hwhite, 3);
    Lxu = expand_lxu(Lxu, U);
    ranges = user_ranges(Lxu);
    R = covariance_cells(info.Rxxs, Lxu, U, N);
    orderings = info.orderings;

    if isempty(info.frac)
        fractions = 1;
    else
        fractions = info.frac(:);
    end

    blended = zeros(U, 1);

    for k = 1:numel(orderings)
        blended = blended + fractions(k) * ...
            oracle_order_rates(Hwhite, R, ranges, orderings{k}, cb);
    end

    gap = max(abs(blended - bu_a(:))) / max(1, max(abs(bu_a)));
end

function rates = oracle_order_rates(Hwhite, R, ranges, ordering, cb)
    U = numel(ranges);
    N = size(Hwhite, 3);
    Ly = size(Hwhite, 1);
    rates = zeros(U, 1);
    stacking = fliplr(ordering(:)');

    for n = 1:N
        accumulated = eye(Ly);
        previousLogdet = 0;

        for k = 1:U
            u = stacking(k);
            Hu = Hwhite(:, ranges{u}, n);
            accumulated = accumulated + Hu * R{u, n} * Hu';
            current = logdet_spd(0.5 * (accumulated + accumulated'));
            rates(u) = rates(u) + (current - previousLogdet) / (cb * log(2));
            previousLogdet = current;
        end

    end

end

function error = learned_closure_error(info)
    total = 0;
    reference = 0;

    for n = 1:numel(info.J)
        G = info.G{n};
        rebuilt = G' * diag(info.S0{n}) * G;
        total = total + norm(info.J{n} - rebuilt, 'fro') ^ 2;
        reference = reference + norm(info.J{n}, 'fro') ^ 2;
    end

    error = sqrt(total / max(reference, realmin));
end

function R = covariance_cells(Rxxs, Lxu, U, N)

    if iscell(Rxxs)
        R = Rxxs;
        return;
    end

    R = cell(U, N);

    for u = 1:U

        for n = 1:N
            R{u, n} = Rxxs(1:Lxu(u), 1:Lxu(u), u, n);
        end

    end

end

function ranges = user_ranges(Lxu)
    ends = cumsum(Lxu);
    starts = [1, ends(1:end - 1) + 1];
    ranges = cell(1, numel(Lxu));

    for u = 1:numel(Lxu)
        ranges{u} = starts(u):ends(u);
    end

end

function value = logdet_spd(matrix)
    [C, flag] = chol(matrix);

    if flag ~= 0
        value = sum(log(max(real(eig(matrix)), realmin)));
        return;
    end

    value = 2 * sum(log(real(diag(C))));
end

function ok = check_psd(Rxxs)
    ok = true;

    if iscell(Rxxs)
        blocks = Rxxs(:);
    else
        blocks = cell(size(Rxxs, 3) * size(Rxxs, 4), 1);
        index = 0;

        for u = 1:size(Rxxs, 3)

            for n = 1:size(Rxxs, 4)
                index = index + 1;
                blocks{index} = Rxxs(:, :, u, n);
            end

        end

    end

    for i = 1:numel(blocks)
        block = 0.5 * (blocks{i} + blocks{i}');

        if min(real(eig(block))) < -1e-9
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
