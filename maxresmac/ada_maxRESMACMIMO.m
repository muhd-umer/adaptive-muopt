function [Rxxs, Eun, w, bun, info] = ada_maxRESMACMIMO(pilot, Lxu, Etotal, theta, cb, opts)
    %ADA_MAXRESMACMIMO Learn a maxRESMAC covariance design from pilot samples
    %
    % Args:
    %   pilot: Structure with numRx, numTones, and sample fields where
    %     sample(n, Atx, X) returns numRx-by-size(X, 2) received samples
    %   Lxu: Number of transmit dimensions per user
    %   Etotal: Total transmit-energy budget
    %   theta: Nonnegative user weights
    %   cb: Rate scaling, one for complex and two for real baseband
    %   opts: Optional algorithm settings
    %
    % Returns:
    %   Rxxs: Per-user, per-tone covariance matrices
    %   Eun: Per-user, per-tone energies
    %   w: Estimated common energy multiplier
    %   bun: Learned rates at the observed covariance
    %   info: Learned receiver quantities and convergence diagnostics, with
    %     observedR identifying the measured covariance in single-pass mode

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    tstart = tic;
    [pilot, Lxu, Etotal, theta, cb, uniformFlag] = validate_inputs( ...
        pilot, Lxu, Etotal, theta, cb);
    U = numel(theta);
    N = pilot.numTones;
    Ly = pilot.numRx;
    Ltot = sum(Lxu);
    [stheta, order] = sort(theta, 'descend');
    delta = stheta - [stheta(2:end); 0];
    ranges = user_ranges(Lxu);
    sortedRanges = user_ranges(Lxu(order));
    sortedRows = concatenate_ranges(ranges(order));

    numPilots = get_opt(opts, 'numPilots', max(256, 8 * (Ltot + Ly)));
    maxOuter = get_opt(opts, 'maxOuter', 100);
    maxBacktrack = get_opt(opts, 'maxBacktrack', 12);
    stepShrink = get_opt(opts, 'stepShrink', 0.5);
    armijo = get_opt(opts, 'armijo', 1e-4);
    ridge = get_opt(opts, 'ridge', 1e-10);
    errorFloor = get_opt(opts, 'errorFloor', 1e-10);
    tolObjective = get_opt(opts, 'tolObjective', 1e-7);
    tolCovariance = get_opt(opts, 'tolCovariance', 1e-6);
    patience = get_opt(opts, 'patience', 3);
    objectiveSlack = get_opt(opts, 'objectiveSlack', 1e-10);
    verbose = get_opt(opts, 'verbose', false);
    rngSeed = get_opt(opts, 'rngSeed', 7);
    singlePass = get_opt(opts, 'singlePass', false);

    if ~isscalar(singlePass) || ...
            ~(islogical(singlePass) || (isnumeric(singlePass) && ismember(singlePass, [0, 1])))
        error('singlePass must be a logical scalar');
    end

    singlePass = logical(singlePass);

    validate_options(numPilots, maxOuter, maxBacktrack, stepShrink, ...
        armijo, ridge, errorFloor, patience);
    rng(rngSeed);
    X = training_pilots(Ltot, numPilots, cb);

    averageEnergy = Etotal / max(N * Ltot, 1);
    floorInitial = get_opt(opts, 'floorInitial', 1e-3 * averageEnergy);
    floorFinal = get_opt(opts, 'floorFinal', 1e-9 * averageEnergy);
    floorInitial = max(real(floorInitial), 0);
    floorFinal = max(min(real(floorFinal), floorInitial), 0);

    if Etotal > 0 && (floorInitial <= 0 || floorFinal <= 0)
        error('positive energy requires positive exploration floors');
    end

    if floorInitial * N * Ltot > Etotal +1e-12
        error('floorInitial is incompatible with Etotal');
    end

    R = initial_covariance(opts, Lxu, N, Etotal, ranges, floorInitial);
    data = evaluate_design(pilot, R, X, ranges, sortedRanges, sortedRows, ...
        order, delta, cb, ridge, errorFloor, [], singlePass);

    if Etotal > 0
        data = attach_gradients(data, sortedRanges, order, delta, cb);
    end

    initialObjective = data.objective;
    objectiveHistory = nan(maxOuter + 1, 1);
    covarianceHistory = nan(maxOuter, 1);
    stepHistory = nan(maxOuter, 1);
    floorHistory = nan(maxOuter, 1);
    objectiveHistory(1) = data.objective;
    accepted = 0;
    stableCount = 0;
    converged = false;
    pilotCalls = N;
    lastFloor = floorInitial;

    if isfield(opts, 'stepInitial') && ~isempty(opts.stepInitial)
        step = real(opts.stepInitial);
    else
        step = 0.5 * max(Etotal, 1);
    end

    if ~isscalar(step) || ~isfinite(step) || step <= 0
        error('stepInitial must be a positive finite scalar');
    end

    if singlePass
        currentR = R;
        [direction, gradientNorm] = normalized_gradient(data.gradient);

        if gradientNorm > eps && Etotal > 0
            tentative = add_direction(R, direction, step);
            R = project_covariances(tentative, ranges, Etotal, floorInitial);
            displacement = covariance_difference(R, currentR);
            covarianceChange = covariance_norm(displacement) / ...
                max(1, covariance_norm(currentR));
            accepted = 1;
        else
            covarianceChange = 0;
        end

        if Etotal == 0
            data.objective = 0;
            data.bun(:) = 0;

            for n = 1:N
                data.Wle{n} = zeros(Ltot, Ly);
                data.W{n} = zeros(Ltot, Ly);
                data.B{n} = zeros(Ltot);
                data.G{n} = eye(Ltot);
                data.S0{n} = ones(Ltot, 1);
                data.J{n} = eye(Ltot);
            end

        end

        [Rxxs, Eun] = format_covariances(R, Lxu, N, uniformFlag);
        bun = data.bun;
        lambda = estimate_multiplier(currentR, data.gradient, floorInitial);
        w = lambda * cb * log(2) * ones(U, 1);
        info = struct();
        info.order = order;
        info.delta = delta;
        info.Wle = data.Wle;
        info.W = data.W;
        info.B = data.B;
        info.G = data.G;
        info.S0 = data.S0;
        info.J = data.J;
        info.gradient = data.gradient;
        info.lambda = lambda;
        info.objective = data.objective;
        info.initialObjective = data.objective;
        info.objectiveHistory = data.objective;
        info.covarianceHistory = covarianceChange;
        info.stepHistory = step;
        info.floorHistory = floorInitial;
        info.iterations = accepted;
        info.converged = false;
        info.pilotCalls = N;
        info.pilotSymbols = N * numPilots;
        info.numPilots = numPilots;
        info.observedR = currentR;
        info.state = struct('R0', {R});
        info.runtime = toc(tstart);
        return;
    end

    alphaMin = get_opt(opts, 'alphaMin', 1e-12);
    alphaMax = get_opt(opts, 'alphaMax', 1e12);
    memory = get_opt(opts, 'nonmonotone', 10);
    floorDecay = get_opt(opts, 'floorDecay', 0.25);

    if ~isscalar(floorDecay) || floorDecay <= 0 || floorDecay >= 1
        error('floorDecay must lie in the open interval (0, 1)');
    end

    if ~isscalar(memory) || memory < 1 || memory ~= fix(memory)
        error('nonmonotone memory must be a positive integer');
    end

    gradientNorm = covariance_norm(data.gradient);
    alpha = min(alphaMax, max(alphaMin, max(Etotal, 1) / max(gradientNorm, eps)));
    objectiveWindow = -inf(memory, 1);
    windowPos = 1;
    objectiveWindow(windowPos) = data.objective;
    previousR = [];
    previousGradient = [];

    for iter = 1:maxOuter
        floorNow = exploration_floor(floorInitial, floorFinal, iter, floorDecay);
        floorHistory(iter) = floorNow;
        lastFloor = floorNow;
        floorBottomed = floorNow <= floorFinal * (1 +1e-9);
        gradientNorm = covariance_norm(data.gradient);

        if gradientNorm <= eps || Etotal == 0
            converged = true;
            break;
        end

        if ~isempty(previousGradient)
            s = covariance_difference(R, previousR);
            y = covariance_difference(previousGradient, data.gradient);
            sy = gradient_inner_product(s, y);

            if sy > 0
                alpha = min(alphaMax, max(alphaMin, covariance_norm(s) ^ 2 / sy));
            end

        end

        tentative = add_direction(R, data.gradient, alpha);
        [projected, projectedFactors] = project_covariances( ...
            tentative, ranges, Etotal, floorNow);
        d = covariance_difference(projected, R);
        directionalGain = gradient_inner_product(data.gradient, d);

        if directionalGain <= objectiveSlack

            if floorBottomed
                converged = true;
                break;
            end

            continue;
        end

        reference = max(objectiveWindow);
        acceptedStep = false;
        candidate = projected;
        candidateData = data;
        trialStep = 1;

        for bt = 1:maxBacktrack

            if trialStep == 1
                candidate = projected;
                candidateFactors = projectedFactors;
            else
                candidate = add_direction(R, d, trialStep);
                candidateFactors = [];
            end

            candidateData = evaluate_design(pilot, candidate, X, ranges, ...
                sortedRanges, sortedRows, order, delta, cb, ridge, errorFloor, ...
                candidateFactors);
            pilotCalls = pilotCalls + N;

            if candidateData.objective >= reference + armijo * trialStep * directionalGain - objectiveSlack
                acceptedStep = true;
                break;
            end

            trialStep = trialStep * stepShrink;
        end

        if ~acceptedStep
            break;
        end

        previousObjective = data.objective;
        previousR = R;
        previousGradient = data.gradient;
        displacement = covariance_difference(candidate, R);
        covarianceChange = covariance_norm(displacement) / max(1, covariance_norm(R));
        R = candidate;
        data = attach_gradients(candidateData, sortedRanges, order, delta, cb);
        accepted = accepted + 1;
        objectiveHistory(accepted + 1) = data.objective;
        covarianceHistory(accepted) = covarianceChange;
        stepHistory(accepted) = trialStep * alpha;
        windowPos = mod(windowPos, memory) + 1;
        objectiveWindow(windowPos) = data.objective;
        relativeChange = abs(data.objective - previousObjective) / max(1, abs(previousObjective));

        if verbose
            fprintf('ada maxresmac iter %d objective %.10g step %.3g change %.3g\n', ...
                iter, data.objective, trialStep * alpha, covarianceChange);
        end

        if relativeChange <= tolObjective || covarianceChange <= tolCovariance
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end

        if stableCount >= patience && floorBottomed
            converged = true;
            break;
        end

    end

    prefinalR = R;
    prefinalGradient = data.gradient;
    R = remove_exploration_floor(R, ranges, Etotal, lastFloor);
    finalData = evaluate_design(pilot, R, X, ranges, sortedRanges, sortedRows, ...
        order, delta, cb, ridge, errorFloor, [], true);
    pilotCalls = pilotCalls + N;

    if Etotal == 0
        initialObjective = 0;
        objectiveHistory(1) = 0;
        finalData.objective = 0;
        finalData.bun(:) = 0;

        for n = 1:N
            finalData.Wle{n} = zeros(Ltot, Ly);
            finalData.W{n} = zeros(Ltot, Ly);
            finalData.B{n} = zeros(Ltot);
            finalData.G{n} = eye(Ltot);
            finalData.S0{n} = ones(Ltot, 1);
            finalData.J{n} = eye(Ltot);
        end

    end

    [Rxxs, Eun] = format_covariances(R, Lxu, N, uniformFlag);
    bun = finalData.bun;
    lambda = estimate_multiplier(prefinalR, prefinalGradient, lastFloor);
    w = lambda * cb * log(2) * ones(U, 1);

    info = struct();
    info.order = order;
    info.delta = delta;
    info.Wle = finalData.Wle;
    info.W = finalData.W;
    info.B = finalData.B;
    info.G = finalData.G;
    info.S0 = finalData.S0;
    info.J = finalData.J;
    info.gradient = prefinalGradient;
    info.lambda = lambda;
    info.objective = finalData.objective;
    info.initialObjective = initialObjective;
    info.objectiveHistory = objectiveHistory(1:accepted + 1);
    info.covarianceHistory = covarianceHistory(1:accepted);
    info.stepHistory = stepHistory(1:accepted);
    info.floorHistory = floorHistory(1:max(accepted, 1));
    info.iterations = accepted;
    info.converged = converged;
    info.pilotCalls = pilotCalls;
    info.pilotSymbols = pilotCalls * numPilots;
    info.numPilots = numPilots;
    info.state = struct('R0', {R});
    info.runtime = toc(tstart);
end

function [pilot, Lxu, Etotal, theta, cb, uniformFlag] = validate_inputs( ...
        pilot, Lxu, Etotal, theta, cb)
    required = {'numRx', 'numTones', 'sample'};

    if ~isstruct(pilot) || ~all(isfield(pilot, required))
        error('pilot must contain numRx, numTones, and sample');
    end

    if ~isscalar(pilot.numRx) || pilot.numRx < 1 || pilot.numRx ~= fix(pilot.numRx)
        error('pilot.numRx must be a positive integer');
    end

    if ~isscalar(pilot.numTones) || pilot.numTones < 1 || ...
            pilot.numTones ~= fix(pilot.numTones)
        error('pilot.numTones must be a positive integer');
    end

    if ~isa(pilot.sample, 'function_handle')
        error('pilot.sample must be a function handle');
    end

    theta = theta(:);

    if isempty(theta) || any(~isfinite(theta)) || any(theta < 0)
        error('theta must contain finite nonnegative weights');
    end

    uniformFlag = isscalar(Lxu);

    if uniformFlag
        Lxu = repmat(Lxu, 1, numel(theta));
    else
        Lxu = reshape(Lxu, 1, []);
    end

    if numel(Lxu) ~= numel(theta) || any(Lxu < 1) || any(Lxu ~= fix(Lxu))
        error('Lxu must contain one positive integer per user');
    end

    Etotal = sum(Etotal(:));

    if ~isscalar(Etotal) || ~isfinite(Etotal) || Etotal < 0
        error('Etotal must be finite and nonnegative');
    end

    if ~isscalar(cb) || ~ismember(cb, [1, 2])
        error('cb must be one or two');
    end

end

function validate_options(numPilots, maxOuter, maxBacktrack, stepShrink, ...
        armijo, ridge, errorFloor, patience)
    integers = [numPilots, maxOuter, maxBacktrack, patience];

    if any(integers < 1) || any(integers ~= fix(integers))
        error('iteration and pilot counts must be positive integers');
    end

    if stepShrink <= 0 || stepShrink >= 1
        error('invalid step shrink factor');
    end

    if armijo < 0 || ridge < 0 || errorFloor <= 0 || errorFloor >= 1
        error('invalid line-search or regularization setting');
    end

end

function X = training_pilots(Ltot, count, cb)

    if count < Ltot
        error('numPilots must be at least sum(Lxu)');
    end

    if cb == 2
        Z = randn(count, Ltot);
    else
        Z = (randn(count, Ltot) + 1j * randn(count, Ltot)) / sqrt(2);
    end

    [Q, ~] = qr(Z, 0);
    X = sqrt(count) * Q';
end

function R = initial_covariance(opts, Lxu, N, Etotal, ranges, floorValue)
    U = numel(Lxu);

    if isfield(opts, 'R0') && ~isempty(opts.R0)
        R = unpack_covariances(opts.R0, Lxu, N);
        R = project_covariances(R, ranges, Etotal, floorValue);
        return;
    end

    R = cell(U, N);
    value = Etotal / max(N * sum(Lxu), 1);

    for u = 1:U

        for n = 1:N
            R{u, n} = value * eye(Lxu(u));
        end

    end

end

function data = evaluate_design(pilot, R, X, ranges, sortedRanges, sortedRows, ...
        order, delta, cb, ridge, errorFloor, factors, needOutputs)

    if nargin < 12
        factors = [];
    end

    if nargin < 13
        needOutputs = false;
    end

    U = numel(ranges);
    N = pilot.numTones;
    Ly = pilot.numRx;
    count = size(X, 2);
    Ltot = size(X, 1);
    data.C = cell(1, N);
    data.Ablocks = cell(1, N);
    data.gradient = cell(U, N);
    data.objective = 0;

    if needOutputs
        data.Wle = cell(1, N);
        data.W = cell(1, N);
        data.B = cell(1, N);
        data.G = cell(1, N);
        data.S0 = cell(1, N);
        data.J = cell(1, N);
        data.bun = zeros(U, N);
    end

    for n = 1:N

        if isempty(factors)
            [Atx, Ablocks] = transmit_factor(R(:, n), ranges, Ltot);
        else
            Atx = factors.Atx{n};
            Ablocks = factors.Ablocks(:, n);
        end

        Y = pilot.sample(n, Atx, X);

        if ~isnumeric(Y) || ~isequal(size(Y), [Ly, count]) || any(~isfinite(Y(:)))
            error('pilot.sample returned an invalid received-sample matrix');
        end

        Xs = X(sortedRows, :);
        Ryy = (Y * Y') / count;
        scale = max(real(trace(Ryy)) / max(Ly, 1), 1);
        Wle = (Xs * Y') / count / (hermitian(Ryy) + ridge * scale * eye(Ly));
        residual = Xs - Wle * Y;
        E = hermitian((residual * residual') / count);
        [Ve, De] = eig(E);
        e = min(max(real(diag(De)), errorFloor), 1);
        E = hermitian(Ve * diag(e) * Ve');
        J = hermitian(E \ eye(Ltot));
        [C, flag] = chol(J);

        if flag ~= 0
            [Vj, Dj] = eig(J);
            j = max(real(diag(Dj)), errorFloor);
            J = hermitian(Vj * diag(j) * Vj');
            C = chol(J);
        end

        diagonal = real(diag(C));
        S0 = diagonal .^ 2;
        data.C{n} = C;
        data.Ablocks{n} = Ablocks;

        if needOutputs
            G = diag(1 ./ diagonal) * C;
            data.Wle{n} = Wle;
            data.W{n} = G * Wle;
            data.B{n} = eye(Ltot) - G;
            data.G{n} = G;
            data.S0{n} = S0;
            data.J{n} = J;
        end

        previousEnd = 0;
        cumulativeLogdet = 0;

        for k = 1:U
            currentEnd = sortedRanges{k}(end);
            streamLogdet = sum(log(max(S0(previousEnd + 1:currentEnd), realmin)));

            if needOutputs
                data.bun(order(k), n) = real(streamLogdet / (cb * log(2)));
            end

            cumulativeLogdet = cumulativeLogdet + streamLogdet;
            data.objective = data.objective + ...
                delta(k) * cumulativeLogdet / (cb * log(2));
            previousEnd = currentEnd;
        end

    end

end

function data = attach_gradients(data, sortedRanges, order, delta, cb)

    for n = 1:numel(data.C)
        data.gradient(:, n) = covariance_gradient(data.C{n}, ...
            data.Ablocks{n}, sortedRanges, order, delta, cb);
    end

end

function gradients = covariance_gradient(C, Ablocks, sortedRanges, order, delta, cb)
    U = numel(order);
    gradients = cell(U, 1);
    errors = cell(U, 1);

    for k = 1:U

        if delta(k) ~= 0
            last = sortedRanges{k}(end);
            Ck = C(1:last, 1:last);
            errors{k} = Ck \ (Ck' \ eye(last));
        end

    end

    for j = 1:U
        idx = sortedRanges{j};
        Lj = numel(idx);
        Q = zeros(Lj);

        for k = j:U

            if delta(k) == 0
                continue;
            end

            Ek = errors{k};
            Q = Q + delta(k) * (eye(Lj) - Ek(idx, idx));
        end

        u = order(j);
        A = Ablocks{u};
        gradient = (A' \ (Q / A)) / (cb * log(2));
        gradients{u} = hermitian(gradient);
    end

end

function [Atx, Ablocks] = transmit_factor(column, ranges, Ltot)
    U = numel(ranges);
    Atx = zeros(Ltot);
    Ablocks = cell(U, 1);

    for u = 1:U
        Ru = hermitian(column{u});
        [V, D] = eig(Ru);
        values = max(real(diag(D)), 0);
        [values, permutation] = sort(values, 'descend');
        V = V(:, permutation);
        A = V * diag(sqrt(values));
        Ablocks{u} = A;
        Atx(ranges{u}, ranges{u}) = A;
    end

end

function [direction, gradientNorm] = normalized_gradient(gradient)
    gradientNorm = covariance_norm(gradient);
    direction = gradient;

    if gradientNorm > 0

        for i = 1:numel(direction)
            direction{i} = direction{i} / gradientNorm;
        end

    end

end

function out = add_direction(R, direction, step)
    out = R;

    for i = 1:numel(R)
        out{i} = hermitian(R{i} + step * direction{i});
    end

end

function displacement = covariance_difference(A, B)
    displacement = A;

    for i = 1:numel(A)
        displacement{i} = A{i} - B{i};
    end

end

function value = gradient_inner_product(gradient, displacement)
    value = 0;

    for i = 1:numel(gradient)
        value = value + real(trace(gradient{i}' * displacement{i}));
    end

end

function value = covariance_norm(blocks)
    value = 0;

    for i = 1:numel(blocks)
        value = value + norm(blocks{i}, 'fro') ^ 2;
    end

    value = sqrt(value);
end

function [R, factors] = project_covariances(R, ranges, total, floorValue)
    U = size(R, 1);
    N = size(R, 2);
    vectors = cell(U, N);
    bases = cell(U, N);
    totalDimensions = sum(cellfun(@(block) size(block, 1), R(:)));
    pooled = zeros(totalDimensions, 1);
    offset = 0;

    for u = 1:U

        for n = 1:N
            [V, D] = eig(hermitian(R{u, n}));
            values = real(diag(D));
            vectors{u, n} = values;
            bases{u, n} = V;
            pooled(offset + (1:numel(values))) = values;
            offset = offset + numel(values);
        end

    end

    if isempty(pooled)
        return;
    end

    shifted = pooled - floorValue;
    budget = total - floorValue * numel(pooled);
    positive = max(shifted, 0);

    if sum(positive) <= budget
        projected = floorValue + positive;
    else
        sorted = sort(shifted, 'descend');
        cumulative = cumsum(sorted);
        indices = (1:numel(sorted))';
        active = find(sorted - (cumulative - budget) ./ indices > 0, 1, 'last');
        threshold = (cumulative(active) - budget) / active;
        projected = floorValue + max(shifted - threshold, 0);
    end

    offset = 0;

    if nargout > 1
        Ltot = ranges{end}(end);
        factors.Atx = cell(1, N);
        factors.Ablocks = cell(U, N);

        for n = 1:N
            factors.Atx{n} = zeros(Ltot);
        end

    end

    for u = 1:U

        for n = 1:N
            count = numel(vectors{u, n});
            values = projected(offset + (1:count));
            V = bases{u, n};
            R{u, n} = hermitian(V * diag(values) * V');

            if nargout > 1
                [factorValues, permutation] = sort(values, 'descend');
                A = V(:, permutation) * diag(sqrt(factorValues));
                factors.Ablocks{u, n} = A;
                factors.Atx{n}(ranges{u}, ranges{u}) = A;
            end

            offset = offset + count;
        end

    end

end

function floorValue = exploration_floor(initial, final, iteration, decay)

    if initial <= final
        floorValue = final;
        return;
    end

    floorValue = max(final, initial * decay ^ (iteration - 1));
end

function R = remove_exploration_floor(R, ranges, total, floorValue)

    if total == 0
        R = project_covariances(R, ranges, 0, 0);
        return;
    end

    used = 0;

    for i = 1:numel(R)
        [V, D] = eig(hermitian(R{i}));
        values = max(real(diag(D)) - floorValue, 0);
        R{i} = hermitian(V * diag(values) * V');
        used = used + sum(values);
    end

    if used > 0

        for i = 1:numel(R)
            R{i} = R{i} * (total / used);
        end

    end

    R = project_covariances(R, ranges, total, 0);
end

function lambda = estimate_multiplier(R, gradient, floorValue)
    valueParts = cell(numel(R), 1);
    scale = max(covariance_norm(R), 1);
    threshold = max(10 * floorValue, 1e-9 * scale);

    for i = 1:numel(R)
        [V, D] = eig(hermitian(R{i}));
        active = real(diag(D)) > threshold;

        if any(active)
            marginal = real(diag(V(:, active)' * gradient{i} * V(:, active)));
            valueParts{i} = marginal;
        end

    end

    values = vertcat(valueParts{:});

    if isempty(values)
        lambda = 0;
    else
        lambda = median(values);
    end

end

function [Rxxs, Eun] = format_covariances(R, Lxu, N, uniformFlag)
    U = numel(Lxu);
    Eun = zeros(U, N);

    for u = 1:U

        for n = 1:N
            Eun(u, n) = real(trace(R{u, n}));
        end

    end

    if ~uniformFlag
        Rxxs = R;
        return;
    end

    L = Lxu(1);
    Rxxs = zeros(L, L, U, N);

    for u = 1:U

        for n = 1:N
            Rxxs(:, :, u, n) = R{u, n};
        end

    end

end

function R = unpack_covariances(input, Lxu, N)
    U = numel(Lxu);

    if iscell(input)

        if ~isequal(size(input), [U, N])
            error('opts.R0 cell dimensions do not match users and tones');
        end

        R = input;
    else

        if numel(unique(Lxu)) ~= 1 || size(input, 3) ~= U || size(input, 4) ~= N
            error('opts.R0 tensor dimensions do not match users and tones');
        end

        R = cell(U, N);

        for u = 1:U

            for n = 1:N
                R{u, n} = input(:, :, u, n);
            end

        end

    end

    for u = 1:U

        for n = 1:N

            if ~isequal(size(R{u, n}), [Lxu(u), Lxu(u)]) || ...
                    any(~isfinite(R{u, n}(:)))
                error('opts.R0 contains an invalid covariance block');
            end

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

function indices = concatenate_ranges(ranges)
    count = sum(cellfun(@numel, ranges));
    indices = zeros(1, count);
    offset = 0;

    for i = 1:numel(ranges)
        current = ranges{i};
        indices(offset + (1:numel(current))) = current;
        offset = offset + numel(current);
    end

end

function matrix = hermitian(matrix)
    matrix = 0.5 * (matrix + matrix');
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
