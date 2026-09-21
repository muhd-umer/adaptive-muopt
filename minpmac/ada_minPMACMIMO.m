function [FEAS_FLAG, bu_a, info] = ada_minPMACMIMO(pilot, Lxu, bu_min, w, cb, opts)
    %ADA_MINPMACMIMO Learn a minimum-power MAC design from pilot samples
    %
    % Args:
    %   pilot: Structure with numRx, numTones, and sample fields where
    %     sample(n, Atx, X) returns numRx-by-size(X, 2) received samples
    %   Lxu: Number of transmit dimensions per user
    %   bu_min: Target user rates in bits per channel use
    %   w: Positive per-user energy weights
    %   cb: Rate scaling, one for complex and two for real baseband
    %   opts: Optional algorithm settings
    %
    % Returns:
    %   FEAS_FLAG: Zero infeasible, one single decoding order, two time-sharing
    %   bu_a: Achieved user rates in bits per channel use
    %   info: Learned covariances, energies, rate prices, time-sharing
    %     fractions, receiver quantities, and convergence diagnostics

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    solverDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(solverDir, 'utils'));

    tstart = tic;
    [pilot, Lxu, bu_min, w, cb, uniformFlag] = validate_inputs( ...
        pilot, Lxu, bu_min, w, cb);
    U = numel(bu_min);
    N = pilot.numTones;
    Ly = pilot.numRx;
    Ltot = sum(Lxu);
    ranges = user_ranges(Lxu);

    numPilots = get_opt(opts, 'numPilots', max(256, 8 * (Ltot + Ly)));
    maxCovariance = get_opt(opts, 'maxCovariance', 100);
    maxRefine = get_opt(opts, 'maxRefine', 20);
    maxPrice = get_opt(opts, 'maxPrice', 120);
    priceGapTol = get_opt(opts, 'priceGapTol', 1e-7);
    levelFraction = get_opt(opts, 'levelFraction', 0.5);
    maxBacktrack = get_opt(opts, 'maxBacktrack', 12);
    stepShrink = get_opt(opts, 'stepShrink', 0.5);
    armijo = get_opt(opts, 'armijo', 1e-4);
    ridge = get_opt(opts, 'ridge', 1e-10);
    errorFloor = get_opt(opts, 'errorFloor', 1e-10);
    tolObjective = get_opt(opts, 'tolObjective', 1e-11);
    tolCovariance = get_opt(opts, 'tolCovariance', 1e-9);
    patience = get_opt(opts, 'patience', 3);
    objectiveSlack = get_opt(opts, 'objectiveSlack', 1e-13);
    shareTol = get_opt(opts, 'shareTol', 1e-9) * max(sum(bu_min), 1);
    clusterTol = get_opt(opts, 'clusterTol', 1e-4);
    maxOrders = get_opt(opts, 'maxOrders', 720);
    maxRestore = get_opt(opts, 'maxRestore', 30);
    restoreRefine = get_opt(opts, 'restoreRefine', 20);
    orderTol = get_opt(opts, 'orderTol', 1e-4) * max(sum(bu_min), 1);
    certTol = get_opt(opts, 'certTol', 1e-7) * max(sum(bu_min), 1);
    verbose = get_opt(opts, 'verbose', false);
    rngSeed = get_opt(opts, 'rngSeed', 7);
    singlePass = get_opt(opts, 'singlePass', false);

    if ~isscalar(singlePass) || ...
            ~(islogical(singlePass) || (isnumeric(singlePass) && ismember(singlePass, [0, 1])))
        error('singlePass must be a logical scalar');
    end

    singlePass = logical(singlePass);

    if exist('linprog', 'file') ~= 2
        error('ada_minPMACMIMO:linprogMissing', ...
            'the rate-price master needs linprog from the Optimization Toolbox');
    end

    validate_options(numPilots, maxCovariance, maxBacktrack, stepShrink, ...
        armijo, ridge, errorFloor, patience);
    rng(rngSeed);
    X = training_pilots(Ltot, numPilots, cb);

    ctx = struct();
    ctx.pilot = pilot;
    ctx.X = X;
    ctx.ranges = {ranges};
    ctx.Lxu = Lxu;
    ctx.w = w;
    ctx.cb = cb;
    ctx.ridge = ridge;
    ctx.errorFloor = errorFloor;
    ctx.maxBacktrack = maxBacktrack;
    ctx.stepShrink = stepShrink;
    ctx.armijo = armijo;
    ctx.patience = patience;
    ctx.tolObjectiveTight = tolObjective;
    ctx.tolCovarianceTight = tolCovariance;
    ctx.objectiveSlackTight = objectiveSlack;
    ctx.tolObjectiveLoose = get_opt(opts, 'tolObjectiveLoose', 1e-7);
    ctx.tolCovarianceLoose = get_opt(opts, 'tolCovarianceLoose', 1e-6);
    ctx.objectiveSlackLoose = get_opt(opts, 'objectiveSlackLoose', 1e-10);
    ctx.tolObjective = tolObjective;
    ctx.tolCovariance = tolCovariance;
    ctx.objectiveSlack = objectiveSlack;
    ctx.alphaMin = get_opt(opts, 'alphaMin', 1e-12);
    ctx.alphaMax = get_opt(opts, 'alphaMax', 1e12);
    ctx.memory = get_opt(opts, 'nonmonotone', 10);
    ctx.floorDecay = get_opt(opts, 'floorDecay', 0.25);
    ctx.verbose = verbose;

    pilotCalls = 0;
    theta = get_opt(opts, 'theta0', []);

    % the probe only exists to seed the rate prices, so a warm theta skips it
    if isempty(theta)
        [gains, probeCalls] = probe_gains(pilot, X, ranges, Ltot, N, ridge, errorFloor);
        pilotCalls = pilotCalls + probeCalls;
        [theta, ceiling] = initial_prices(gains, bu_min, w, cb);
        energyScale = price_energy_scale(theta, w, cb, gains);
    else
        theta = max(real(theta(:)), 0);

        if numel(theta) ~= U
            error('opts.theta0 must contain one nonnegative price per user');
        end

        ceiling = 2 * theta;
        energyScale = price_energy_scale(theta, w, cb, 1);
    end

    thetaBox = get_opt(opts, 'thetaMax', max(ceiling(:), 1e-6)) .* ones(U, 1);
    floorInitial = get_opt(opts, 'floorInitial', 1e-4 * energyScale);
    floorFinal = get_opt(opts, 'floorFinal', 1e-10 * energyScale);
    floorInitial = max(real(floorInitial), 0);
    floorFinal = max(min(real(floorFinal), floorInitial), 0);

    if isfield(opts, 'R0') && ~isempty(opts.R0)
        R = unpack_covariances(opts.R0, Lxu, N);
        R = project_covariances(R, floorFinal);
    else
        R = initial_covariance(Lxu, N, floorInitial);
    end

    if singlePass
        [R, data, calls, observedR] = tracking_step(ctx, R, theta, floorFinal, ...
            get_opt(opts, 'stepInitial', 0.5 * energyScale));
        pilotCalls = pilotCalls + calls;
        [Rxxs, Eun] = format_covariances(R, Lxu, N, uniformFlag);
        bu_a = sum(data.bun, 2)';
        FEAS_FLAG = double(all(bu_a(:) >= bu_min(:) - shareTol));
        info = base_info(data, theta, Rxxs, Eun, Lxu, N, w);
        info.frac = [];
        info.orderings = {price_order(theta)};
        info.clusters = {};
        info.bun = data.bun;
        info.priceIterations = 0;
        info.priceResidual = norm(sum(data.bun, 2) - bu_min);
        info.converged = false;
        info.iterations = 1;
        info.observedR = observedR;
        info.pilotCalls = pilotCalls;
        info.pilotSymbols = pilotCalls * numPilots;
        info.numPilots = numPilots;
        info.state = struct('R0', {R});
        info.runtime = toc(tstart);
        return;
    end

    priceHistory = nan(U, maxPrice + 1);
    residualHistory = nan(maxPrice + 1, 1);
    gapHistory = nan(maxPrice + 1, 1);
    pool = empty_pool(U);
    spg = [];
    meas = [];
    lambda = [];
    activeColumns = [];
    upperBound = inf;
    lowerBound = -inf;
    dualGap = inf;
    priceConverged = false;
    certificate = false;
    priceIterations = 0;
    thetaCap = thetaBox * 1e12;
    boundTrusted = true;

    for iteration = 1:maxPrice

        if iteration == 1
            budget = maxCovariance;
            startFloor = floorInitial;
        else
            budget = maxRefine;
            startFloor = floorFinal;
        end

        if boundTrusted
            ctx = schedule_tolerances(ctx, abs(dualGap));
        else
            ctx = schedule_tolerances(ctx, 0);
        end

        [R, data, meas, spg, calls] = maximize_lagrangian(ctx, R, meas, spg, theta, ...
            startFloor, floorFinal, budget);
        pilotCalls = pilotCalls + calls;

        rates = sum(data.bun, 2);
        subgradient = rates - bu_min;
        priceHistory(:, iteration) = theta;
        residualHistory(iteration) = norm(subgradient);
        priceIterations = iteration;
        % the measured design is an affine lower bound on the inner maximum for
        % every price vector, so it becomes a cut at no pilot cost
        pool = add_columns(pool, data, R, ranges, Lxu, cb, theta, clusterTol, ...
            maxOrders);

        % only a stationary inner solve bounds the dual from below; a
        % budget-limited one understates the inner maximum and would overstate
        % the bound
        if spg.stationary
            lowerBound = max(lowerBound, real(theta' * bu_min - data.lagrangian));
        end

        [nextTheta, value, nextLambda, nextActive, dualsValid] = ...
            master_prices(pool, bu_min, thetaBox);

        if dualsValid

            if value <= upperBound
                upperBound = value;
                lambda = nextLambda;
                activeColumns = nextActive;
            end

            dualGap = (upperBound - lowerBound) / max(1, abs(upperBound));

            % the upper bound is achievable, so a lower bound above it can only
            % come from an inner solve that stopped short; discard the bound and
            % rebuild it from tighter solves
            if dualGap < -priceGapTol
                lowerBound = -inf;
                dualGap = inf;
                boundTrusted = false;
            end

            if isfinite(lowerBound) && value > lowerBound
                stepped = level_prices(pool, bu_min, thetaBox, theta, ...
                    lowerBound + levelFraction * (value - lowerBound));

                if ~isempty(stepped)
                    nextTheta = stepped;
                end

            end

        elseif any(nextTheta >= 0.999 * thetaBox)
            % the model is still unbounded along some price direction, so the
            % trust region has to make room before its duals mean anything
            thetaBox = min(thetaBox * 10, thetaCap);
        end

        gapHistory(iteration) = dualGap;

        % exact KKT certificate: targets met and every priced user tight
        shortfall = max(max(bu_min - rates), 0);
        slackness = max(theta .* max(rates - bu_min, 0));
        certificate = shortfall <= certTol && slackness <= certTol;

        if verbose
            fprintf('ada minpmac price %d gap %.3e residual %.3e\n', ...
                iteration, dualGap, residualHistory(iteration));
        end

        if certificate || (dualsValid && abs(dualGap) <= priceGapTol)
            priceConverged = true;
            break;
        end

        theta = nextTheta;
    end

    if ~priceConverged
        warning('ada_minPMACMIMO:priceNotConverged', ...
            'rate prices did not converge within %d iterations', priceIterations);
    end

    % the blend of the active designs costs exactly the model optimum, and
    % concavity of logdet keeps every subset capacity above the blended one, so
    % the rate targets survive the mixture
    if ~certificate && ~isempty(activeColumns)
        R = mix_columns(pool, lambda, activeColumns);
    end

    R = remove_exploration_floor(R, floorFinal);
    [order, delta, sortedRanges, sortedRows] = price_sort(theta, ranges, Lxu);
    [data, meas] = evaluate_design(ctx, R, order, delta, sortedRanges, sortedRows, ...
        [], true);
    pilotCalls = pilotCalls + meas.calls;

    [clusters, thetaUnique] = cluster_prices(theta, clusterTol);
    orders = candidate_orders(clusters, U, maxOrders);
    sortedIdx = order_index(orders, fliplr(order(:)'));
    % stripping the floor can leave the design a hair short of the target sum
    % rate, so walk the covariance up until the measured polymatroid contains it
    [R, data, restoreScale, restoreCalls] = restore_feasibility(ctx, R, data, ...
        order, delta, sortedRanges, sortedRows, ranges, bu_min, shareTol, false, ...
        maxRestore, restoreRefine);
    pilotCalls = pilotCalls + restoreCalls;

    % minimum power puts the design on the polymatroid face, and near-degenerate
    % prices leave the price-induced order a fraction of a bit short. The
    % recovery then reports a time-share carrying a ~1e-5 weight on a second
    % order, which is noise rather than a schedule anyone would run. Nudging the
    % design until that one order clears costs power of the same relative order
    % and keeps the reported decoding structure honest. A genuine tie misses by
    % far more than orderTol and still time-shares
    shortfall = -min(sum(data.bun, 2) - bu_min);

    if shortfall > 0 && shortfall <= orderTol
        [R, data, orderScale, orderCalls] = restore_feasibility(ctx, R, data, ...
            order, delta, sortedRanges, sortedRows, ranges, bu_min, shareTol, ...
            true, maxRestore, restoreRefine);
        pilotCalls = pilotCalls + orderCalls;
        restoreScale = restoreScale * orderScale;
    end

    [vertices, perTone] = measured_vertices(data.Jnat, orders, ranges, Lxu, cb);
    recovery = recover_rates(vertices, perTone, orders, sortedIdx, bu_min, ...
        shareTol, U, N);
    FEAS_FLAG = recovery.flag;
    bu_a = recovery.bu_a';

    [Rxxs, Eun] = format_covariances(R, Lxu, N, uniformFlag);
    info = base_info(data, theta, Rxxs, Eun, Lxu, N, w);
    info.clusters = clusters;
    info.thetaUnique = thetaUnique;
    info.restoreScale = restoreScale;
    info.frac = recovery.frac;
    info.orderings = recovery.orderings;
    info.bun = recovery.bun;

    if recovery.flag == 0
        info.feasible = false;
        info.sol_status = 'Failed';
    end

    info.bu_vertices = vertices;
    info.priceIterations = priceIterations;
    info.priceResidual = residualHistory(priceIterations);
    info.priceHistory = priceHistory(:, 1:priceIterations);
    info.residualHistory = residualHistory(1:priceIterations);
    info.gapHistory = gapHistory(1:priceIterations);
    info.dualGap = dualGap;
    info.upperBound = upperBound;
    info.lowerBound = lowerBound;
    info.columns = size(pool.B, 2);
    info.certificate = certificate;
    info.converged = priceConverged;
    info.iterations = priceIterations;
    info.pilotCalls = pilotCalls;
    info.pilotSymbols = pilotCalls * numPilots;
    info.numPilots = numPilots;
    info.state = struct('R0', {R});
    info.runtime = toc(tstart);
end

function info = base_info(data, theta, Rxxs, Eun, Lxu, N, w)
    info = struct();
    info.Lxu = Lxu;
    info.w = w;
    info.theta = theta;
    info.order = data.order;
    info.delta = data.delta;
    info.Rxxs = Rxxs;
    info.Rxx = {Rxxs};
    info.Eun = Eun;
    info.Eu_avg = sum(Eun, 2).' / N;
    info.weightedPower = real(w(:).' * sum(Eun, 2));
    info.Wle = data.Wle;
    info.W = data.W;
    info.B = data.B;
    info.G = data.G;
    info.S0 = data.S0;
    info.J = data.J;
    info.gradient = data.gradient;
    info.objective = data.lagrangian;
    info.weightedRate = data.objective;
    info.learnedRates = data.bun;
    info.feasible = true;
    info.sol_status = 'Solved';
end

function recovery = recover_rates(vertices, perTone, orders, sortedIdx, ...
        bu_min, shareTol, U, N)
    %RECOVER_RATES Pick a decoding order or a time sharing that meets the targets
    %
    % Args:
    %   vertices: Rate vector achieved by each candidate decoding order
    %   perTone: Per-tone rates behind each vertex
    %   orders: Candidate decoding orders
    %   sortedIdx: Row of the price-induced order, zero when absent
    %   bu_min: Target user rates
    %   shareTol: Slack allowed on each target
    %   U: Number of users
    %   N: Number of tones
    %
    % Returns:
    %   recovery: Feasibility flag, achieved rates, fractions, orders, per-tone rates

    recovery = struct('flag', 0, 'bu_a', zeros(U, 1), 'frac', [], ...
        'orderings', {{}}, 'bun', zeros(U, N));

    % a single decoding order suffices whenever it already clears every target;
    % otherwise the optimum sits on a polymatroid face and needs time-sharing
    if sortedIdx > 0 && all(vertices(:, sortedIdx) >= bu_min - shareTol)
        recovery.flag = 1;
        recovery.bu_a = vertices(:, sortedIdx);
        recovery.orderings = {orders(sortedIdx, :)};
        recovery.bun = perTone{sortedIdx};
        return;
    end

    [weights, activeIdx] = solve_time_sharing(vertices, bu_min, shareTol);

    if isempty(weights)
        fallback = max(sortedIdx, 1);
        recovery.bu_a = vertices(:, fallback);
        recovery.orderings = {orders(fallback, :)};
        recovery.bun = perTone{fallback};
        return;
    end

    weights = weights(:);
    activeList = find(activeIdx);
    recovery.bu_a = vertices(:, activeList) * weights;
    blended = zeros(U, N);

    for k = 1:numel(activeList)
        blended = blended + perTone{activeList(k)} * weights(k);
    end

    recovery.flag = 1 + double(numel(activeList) > 1);
    recovery.frac = weights;
    recovery.orderings = num2cell(orders(activeList, :), 2);
    recovery.bun = blended;
end

function index = order_index(orders, target)
    match = all(orders == target, 2);
    index = find(match, 1);

    if isempty(index)
        index = 0;
    end

end

function ok = restore_satisfied(data, ranges, cb, bu_min, margin, useOrderRates)
    %RESTORE_SATISFIED Whether a design clears the targets under the wanted criterion
    %
    % The polymatroid test is the feasibility question: can any time sharing of
    % decoding orders hit the targets. The order test is stricter and asks
    % whether the single price-induced order already does

    if useOrderRates
        ok = min(sum(data.bun, 2) - bu_min) >= margin;
        return;
    end

    ok = polymatroid_slack(data.Jnat, ranges, cb, bu_min) >= margin;
end

function slack = polymatroid_slack(Jnat, ranges, cb, bu_min)
    %POLYMATROID_SLACK Smallest subset margin of bu_min inside the measured region
    U = numel(ranges);
    N = numel(Jnat);
    slack = inf;

    for mask = 1:(2 ^ U - 1)
        members = find(bitget(mask, 1:U));
        rows = concatenate_ranges(ranges(members));
        capacity = 0;

        for n = 1:N
            capacity = capacity + logdet_spd(hermitian(Jnat{n}(rows, rows)));
        end

        capacity = capacity / (cb * log(2));
        slack = min(slack, capacity - sum(bu_min(members)));
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

function scaled = scale_covariances(R, factor)
    scaled = R;

    for i = 1:numel(R)
        scaled{i} = R{i} * factor;
    end

end

function [R, data, factor, calls] = restore_feasibility(ctx, R, data, order, ...
        delta, sortedRanges, sortedRows, ranges, bu_min, margin, useOrderRates, ...
        maxRestore, refine)
    %RESTORE_FEASIBILITY Scale the covariances up until a rate criterion holds
    %
    % Args:
    %   ctx: Solver context
    %   R: Covariance cells to scale
    %   data: Reduced design at R
    %   order: Users sorted by descending price
    %   delta: Successive price differences
    %   sortedRanges: Stream index ranges in price order
    %   sortedRows: Natural stream indices in price order
    %   ranges: Natural stream index ranges per user
    %   bu_min: Target user rates
    %   margin: Slack the criterion must clear
    %   useOrderRates: Drive the price-order rates rather than the polymatroid
    %   maxRestore: Expansion attempts
    %   refine: Bisection steps
    %
    % Returns:
    %   R: Scaled covariance cells
    %   data: Reduced design at the scaled covariances
    %   factor: Scale applied
    %   calls: Pilot sweeps consumed

    calls = 0;
    factor = 1;

    if restore_satisfied(data, ranges, ctx.cb, bu_min, margin, useOrderRates)
        return;
    end

    base = R;
    low = 1;
    high = 1;
    found = false;
    increment = 1e-7;

    for attempt = 1:maxRestore
        low = high;
        high = high * (1 + increment);
        increment = increment * 2;
        candidate = scale_covariances(base, high);
        [trial, trialMeas] = evaluate_design(ctx, candidate, order, delta, ...
            sortedRanges, sortedRows, [], true);
        calls = calls + trialMeas.calls;

        if restore_satisfied(trial, ranges, ctx.cb, bu_min, margin, useOrderRates)
            found = true;
            R = candidate;
            data = trial;
            factor = high;
            break;
        end

    end

    if ~found
        return;
    end

    for step = 1:refine
        middle = 0.5 * (low + high);
        candidate = scale_covariances(base, middle);
        [trial, trialMeas] = evaluate_design(ctx, candidate, order, delta, ...
            sortedRanges, sortedRows, [], true);
        calls = calls + trialMeas.calls;

        if restore_satisfied(trial, ranges, ctx.cb, bu_min, margin, useOrderRates)
            high = middle;
            R = candidate;
            data = trial;
            factor = middle;
        else
            low = middle;
        end

    end

end

function pool = empty_pool(U)
    pool = struct('B', zeros(U, 0), 'E', zeros(1, 0), 'index', zeros(1, 0), 'R', {{}});
end

function pool = add_columns(pool, data, R, ranges, Lxu, cb, theta, clusterTol, maxOrders)
    %ADD_COLUMNS Record the rate vertices a measured design contributes to the dual model
    %
    % For fixed covariances and a fixed decoding order the inner objective is
    % affine in theta, so every design already measured is a valid cut at no
    % extra pilot cost
    %
    % Args:
    %   pool: Existing cut pool
    %   data: Reduced design carrying bun, energy, and Jnat
    %   R: Covariance cells that generated the design
    %   ranges: Natural stream index ranges per user
    %   Lxu: Transmit dimensions per user
    %   cb: Rate scaling
    %   theta: Prices that produced the design
    %   clusterTol: Tolerance for treating prices as tied
    %   maxOrders: Cap on enumerated tied orders
    %
    % Returns:
    %   pool: Pool extended with one column per contributed vertex

    vertices = sum(data.bun, 2);
    clusters = cluster_prices(theta, clusterTol);

    % tied prices leave the decoding order undetermined, and each tied order is
    % a distinct vertex of the same measured polymatroid
    if numel(clusters) < numel(theta)

        try
            orders = decoding_orders(clusters, maxOrders);
            extra = measured_vertices(data.Jnat, orders, ranges, Lxu, cb);
            vertices = [vertices, extra];
        catch
        end

    end

    pool.R{end + 1} = R;
    source = numel(pool.R);

    for k = 1:size(vertices, 2)
        pool.B(:, end + 1) = vertices(:, k);
        pool.E(end + 1) = data.energy;
        pool.index(end + 1) = source;
    end

end

function [theta, value, lambda, active, valid] = master_prices(pool, bu_min, box)
    %MASTER_PRICES Maximize the cutting-plane model of the dual over nonnegative prices
    %
    % The model is max_theta bu_min'theta - z subject to z >= B'theta - E. Its
    % linear-program duals solve min E'lambda subject to B lambda >= bu_min and
    % sum(lambda) = 1, which is exactly the time-sharing problem over the
    % designs measured so far
    %
    % Args:
    %   pool: Cut pool of rate vertices and weighted energies
    %   bu_min: Target user rates
    %   box: Per-user upper bound acting as a trust region
    %
    % Returns:
    %   theta: Model-optimal prices, usable as the next query point
    %   value: Model optimum, an upper bound on the optimal weighted power
    %   lambda: Normalized time-sharing weights over the pooled columns
    %   active: Indices of columns carrying weight
    %   valid: True when the trust region is slack so the duals are meaningful

    U = numel(bu_min);
    K = size(pool.B, 2);
    value = inf;
    lambda = [];
    active = [];
    valid = false;

    objective = [-bu_min(:); 1];
    inequality = [pool.B', -ones(K, 1)];
    bounds = real(pool.E(:));
    lower = [zeros(U, 1); -inf];
    upper = [box(:); inf];
    options = optimoptions('linprog', 'Display', 'none');
    [solution, objectiveValue, exitflag, ~, multipliers] = linprog( ...
        objective, inequality, bounds, [], [], lower, upper, options);

    if exitflag ~= 1 || isempty(solution)
        error('ada_minPMACMIMO:masterFailed', ...
            'the rate-price master linear program failed with exit flag %d', exitflag);
    end

    theta = max(real(solution(1:U)), 0);
    weights = max(real(multipliers.ineqlin(:)), 0);
    total = sum(weights);

    % a binding trust region means the model is still unbounded in some price
    % direction, so the duals do not yet describe a feasible time sharing
    if total <= 0 || any(theta >= 0.999 * box(:))
        return;
    end

    value = -objectiveValue;
    lambda = weights / total;
    active = find(lambda > 1e-9);
    valid = ~isempty(active) && all(pool.B(:, active) * lambda(active) >= ...
        bu_min(:) - 1e-6 * max(1, max(abs(bu_min))));
end

function theta = level_prices(pool, bu_min, box, center, level)
    %LEVEL_PRICES Nearest point to the centre whose model value clears a level
    %
    % Plain cutting planes chase the model maximum and oscillate between
    % extreme prices. Constraining the model value to a level between the two
    % bounds and then staying as close as possible to the incumbent damps that
    % oscillation while still guaranteeing progress
    %
    % Args:
    %   pool: Cut pool of rate vertices and weighted energies
    %   bu_min: Target user rates
    %   box: Per-user upper bound on the prices
    %   center: Incumbent prices to stay near
    %   level: Model value the next iterate must reach
    %
    % Returns:
    %   theta: Next query prices, empty when the level program is infeasible

    U = numel(bu_min);
    K = size(pool.B, 2);
    theta = [];
    identity = eye(U);
    zerosU = zeros(U, 1);

    % variables are the prices, the model epigraph value, and the L-infinity
    % distance to the centre
    objective = [zerosU; 0; 1];
    inequality = [pool.B', -ones(K, 1), zeros(K, 1); ...
        -bu_min(:)', 1, 0; ...
        identity, zerosU, -ones(U, 1); ...
        -identity, zerosU, -ones(U, 1)];
    bounds = [real(pool.E(:)); -level; center(:); -center(:)];
    lower = [zerosU; -inf; 0];
    upper = [box(:); inf; inf];
    options = optimoptions('linprog', 'Display', 'none');
    [solution, ~, exitflag] = linprog(objective, inequality, bounds, [], [], ...
        lower, upper, options);

    if exitflag ~= 1 || isempty(solution)
        return;
    end

    theta = max(real(solution(1:U)), 0);
end

function R = mix_columns(pool, lambda, active)
    %MIX_COLUMNS Blend the active designs into a single feasible covariance
    %
    % Weighted energy is linear in R, so the blend costs exactly the model
    % optimum, and logdet is concave in R, so every subset capacity of the blend
    % dominates the blended capacities and the rate targets stay satisfied

    R = pool.R{pool.index(active(1))};
    weight = lambda(active(1));

    for i = 1:numel(R)
        R{i} = weight * R{i};
    end

    for k = 2:numel(active)
        source = pool.R{pool.index(active(k))};
        weight = lambda(active(k));

        for i = 1:numel(R)
            R{i} = R{i} + weight * source{i};
        end

    end

    for i = 1:numel(R)
        R{i} = hermitian(R{i});
    end

end

function ctx = schedule_tolerances(ctx, gap)
    %SCHEDULE_TOLERANCES Loosen the inner solve while the certified dual gap is wide
    %
    % The outer model only consumes a supergradient, so solving the inner
    % problem to final precision before the prices have settled is wasted work

    if isnan(gap) || ~isfinite(gap)
        target = inf;
    elseif gap <= 0
        target = 0;
    else
        target = 1e-3 * gap;
    end

    ctx.tolObjective = max(ctx.tolObjectiveTight, ...
        min(ctx.tolObjectiveLoose, target));
    ctx.tolCovariance = max(ctx.tolCovarianceTight, ...
        min(ctx.tolCovarianceLoose, target));
    ctx.objectiveSlack = max(ctx.objectiveSlackTight, ...
        min(ctx.objectiveSlackLoose, target));
end

function orders = candidate_orders(clusters, U, maxOrders)
    %CANDIDATE_ORDERS Decoding orders the recovered design may time-share over
    %
    % Distinct prices pin the order the dual prefers, but the optimum can still
    % sit on a polymatroid face that no single order reaches, so the full
    % permutation set stays in play whenever it is affordable

    if factorial(U) <= maxOrders
        orders = perms(1:U);
        return;
    end

    orders = decoding_orders(clusters, maxOrders);
end

function [R, data, meas, spg, calls] = maximize_lagrangian(ctx, R, meas, spg, ...
        theta, startFloor, finalFloor, maxIters)
    %MAXIMIZE_LAGRANGIAN Spectral projected gradient ascent on the priced Lagrangian
    %
    % Args:
    %   ctx: Solver context
    %   R: Warm-start covariance cells
    %   meas: Cached measurement at R, or empty to measure
    %   spg: Carried step-size state, or empty on the first call
    %   theta: Current rate prices
    %   startFloor: Initial exploration floor
    %   finalFloor: Terminal exploration floor
    %   maxIters: Inner iteration budget
    %
    % Returns:
    %   R: Updated covariance cells
    %   data: Reduced design at the returned R
    %   meas: Measurement at the returned R
    %   spg: Updated step-size state
    %   calls: Pilot sweeps consumed

    ranges = ctx.ranges{1};
    [order, delta, sortedRanges, sortedRows] = price_sort(theta, ranges, ctx.Lxu);
    calls = 0;
    stationary = false;

    % iterations past the first re-enter at the terminal floor with R already
    % floored, so the cached measurement is still valid and costs nothing
    if isempty(meas) || startFloor > finalFloor * (1 + 1e-9)
        R = project_covariances(R, startFloor);
        meas = measure_design(ctx, R, []);
        calls = calls + meas.calls;
    end

    data = reduce_design(ctx, meas, R, order, delta, sortedRanges, sortedRows, false);
    data = attach_gradients(ctx, data, R, sortedRanges, order, delta);

    % the step scale survives a price change; the secant pair and the
    % nonmonotone reference do not, because the objective itself moved
    if isempty(spg) || ~isfield(spg, 'alpha') || ~isfinite(spg.alpha)
        gradientNorm = covariance_norm(data.gradient);
        scale = max(covariance_norm(R), 1);
        spg = struct('alpha', min(ctx.alphaMax, max(ctx.alphaMin, ...
            scale / max(gradientNorm, eps))), 'stationary', false);
    end

    alpha = spg.alpha;
    objectiveWindow = -inf(ctx.memory, 1);
    windowPos = 1;
    objectiveWindow(windowPos) = data.lagrangian;
    previousR = [];
    previousGradient = [];
    stableCount = 0;

    for iter = 1:maxIters
        floorNow = max(finalFloor, startFloor * ctx.floorDecay ^ (iter - 1));
        floorBottomed = floorNow <= finalFloor * (1 + 1e-9);
        gradientNorm = covariance_norm(data.gradient);

        if gradientNorm <= eps
            stationary = true;
            break;
        end

        if ~isempty(previousGradient)
            s = covariance_difference(R, previousR);
            y = covariance_difference(previousGradient, data.gradient);
            sy = gradient_inner_product(s, y);

            if sy > 0
                alpha = min(ctx.alphaMax, max(ctx.alphaMin, ...
                    covariance_norm(s) ^ 2 / sy));
            end

        end

        tentative = add_direction(R, data.gradient, alpha);
        [projected, projectedFactors] = project_covariances(tentative, floorNow, ctx.ranges{1});
        d = covariance_difference(projected, R);
        directionalGain = gradient_inner_product(data.gradient, d);

        if directionalGain <= ctx.objectiveSlack

            if floorBottomed
                stationary = true;
                break;
            end

            continue;
        end

        reference = max(objectiveWindow);
        acceptedStep = false;
        trialStep = 1;
        candidate = projected;
        candidateData = data;
        candidateMeas = meas;

        for bt = 1:ctx.maxBacktrack

            if trialStep == 1
                candidate = projected;
                candidateFactors = projectedFactors;
            else
                candidate = add_direction(R, d, trialStep);
                candidateFactors = [];
            end

            candidateMeas = measure_design(ctx, candidate, candidateFactors);
            calls = calls + candidateMeas.calls;
            candidateData = reduce_design(ctx, candidateMeas, candidate, order, ...
                delta, sortedRanges, sortedRows, false);

            if candidateData.lagrangian >= reference + ...
                    ctx.armijo * trialStep * directionalGain - ctx.objectiveSlack
                acceptedStep = true;
                break;
            end

            trialStep = trialStep * ctx.stepShrink;
        end

        if ~acceptedStep
            stationary = floorBottomed;
            break;
        end

        previousObjective = data.lagrangian;
        previousR = R;
        previousGradient = data.gradient;
        displacement = covariance_difference(candidate, R);
        covarianceChange = covariance_norm(displacement) / max(1, covariance_norm(R));
        R = candidate;
        meas = candidateMeas;
        data = attach_gradients(ctx, candidateData, R, sortedRanges, order, delta);
        windowPos = mod(windowPos, ctx.memory) + 1;
        objectiveWindow(windowPos) = data.lagrangian;
        relativeChange = abs(data.lagrangian - previousObjective) / ...
            max(1, abs(previousObjective));

        if ctx.verbose
            fprintf('  ada minpmac inner %d lagrangian %.10g change %.3g\n', ...
                iter, data.lagrangian, covarianceChange);
        end

        if relativeChange <= ctx.tolObjective || covarianceChange <= ctx.tolCovariance
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end

        if stableCount >= ctx.patience && floorBottomed
            stationary = true;
            break;
        end

    end

    spg.alpha = alpha;
    spg.stationary = stationary;
    data = reduce_design(ctx, meas, R, order, delta, sortedRanges, sortedRows, true);
    data = attach_gradients(ctx, data, R, sortedRanges, order, delta);
end

function [R, data, calls, observedR] = tracking_step(ctx, R, theta, floorValue, step)
    ranges = ctx.ranges{1};
    [order, delta, sortedRanges, sortedRows] = price_sort(theta, ranges, ctx.Lxu);
    R = project_covariances(R, floorValue);
    [data, meas] = evaluate_design(ctx, R, order, delta, sortedRanges, sortedRows, ...
        [], true);
    calls = meas.calls;
    data = attach_gradients(ctx, data, R, sortedRanges, order, delta);
    observedR = R;
    [direction, gradientNorm] = normalized_gradient(data.gradient);

    if gradientNorm > eps
        R = project_covariances(add_direction(R, direction, step), floorValue);
    end

end

function [order, delta, sortedRanges, sortedRows] = price_sort(theta, ranges, Lxu)
    [sorted, order] = sort(theta, 'descend');
    delta = sorted - [sorted(2:end); 0];
    sortedRanges = user_ranges(Lxu(order));
    sortedRows = concatenate_ranges(ranges(order));
end

function order = price_order(theta)
    [~, order] = sort(theta, 'descend');
    order = order(:)';
end

function [data, meas] = evaluate_design(ctx, R, order, delta, sortedRanges, ...
        sortedRows, factors, needOutputs)
    meas = measure_design(ctx, R, factors);
    data = reduce_design(ctx, meas, R, order, delta, sortedRanges, sortedRows, ...
        needOutputs);
end

function meas = measure_design(ctx, R, factors)
    %MEASURE_DESIGN Transmit pilots at R and learn the natural-order information matrix
    %
    % Args:
    %   ctx: Solver context holding the pilot interface and regularization
    %   R: Per-user, per-tone transmit covariance cells
    %   factors: Cached transmit factors, or empty to refactor R
    %
    % Returns:
    %   meas: Natural-order Jnat, LMMSE matrices, transmit blocks, and call count

    ranges = ctx.ranges{1};
    pilot = ctx.pilot;
    X = ctx.X;
    N = pilot.numTones;
    Ly = pilot.numRx;
    count = size(X, 2);
    Ltot = size(X, 1);
    meas = struct();
    meas.Jnat = cell(1, N);
    meas.Wle = cell(1, N);
    meas.Ablocks = cell(1, N);
    meas.calls = N;

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

        Ryy = (Y * Y') / count;
        scale = max(real(trace(Ryy)) / max(Ly, 1), 1);
        Wle = (X * Y') / count / (hermitian(Ryy) + ctx.ridge * scale * eye(Ly));
        residual = X - Wle * Y;
        E = hermitian((residual * residual') / count);
        [Ve, De] = eig(E);
        e = min(max(real(diag(De)), ctx.errorFloor), 1);
        E = hermitian(Ve * diag(e) * Ve');
        meas.Jnat{n} = hermitian(E \ eye(Ltot));
        meas.Wle{n} = Wle;
        meas.Ablocks{n} = Ablocks;
    end

end

function data = reduce_design(ctx, meas, R, order, delta, sortedRanges, sortedRows, ...
        needOutputs)
    %REDUCE_DESIGN Turn a natural-order measurement into priced rates and receivers
    %
    % The measurement depends only on R, so a price change is pure linear algebra
    % on the cached Jnat and costs no pilot symbols
    %
    % Args:
    %   ctx: Solver context
    %   meas: Output of measure_design at the same R
    %   R: Per-user, per-tone transmit covariance cells
    %   order: Users sorted by descending price
    %   delta: Successive price differences
    %   sortedRanges: Stream index ranges in price order
    %   sortedRows: Natural stream indices in price order
    %   needOutputs: Also build the GDFE receiver quantities
    %
    % Returns:
    %   data: Rates, weighted objective, Lagrangian, and receiver quantities

    U = numel(order);
    N = numel(meas.Jnat);
    Ltot = numel(sortedRows);
    cb = ctx.cb;
    data = struct();
    data.C = cell(1, N);
    data.Ablocks = meas.Ablocks;
    data.Jnat = meas.Jnat;
    data.gradient = cell(U, N);
    data.objective = 0;
    data.order = order;
    data.delta = delta;
    data.bun = zeros(U, N);

    if needOutputs
        data.Wle = cell(1, N);
        data.W = cell(1, N);
        data.B = cell(1, N);
        data.G = cell(1, N);
        data.S0 = cell(1, N);
        data.J = cell(1, N);
    end

    for n = 1:N
        J = hermitian(meas.Jnat{n}(sortedRows, sortedRows));
        [C, flag] = chol(J);

        if flag ~= 0
            [Vj, Dj] = eig(J);
            j = max(real(diag(Dj)), ctx.errorFloor);
            J = hermitian(Vj * diag(j) * Vj');
            C = chol(J);
        end

        diagonal = real(diag(C));
        S0 = diagonal .^ 2;
        data.C{n} = C;

        if needOutputs
            G = diag(1 ./ diagonal) * C;
            Wle = meas.Wle{n}(sortedRows, :);
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
            data.bun(order(k), n) = real(streamLogdet / (cb * log(2)));
            cumulativeLogdet = cumulativeLogdet + streamLogdet;
            data.objective = data.objective + ...
                delta(k) * cumulativeLogdet / (cb * log(2));
            previousEnd = currentEnd;
        end

    end

    data.energy = weighted_energy(R, ctx.w);
    data.lagrangian = data.objective - data.energy;
end

function value = weighted_energy(R, w)
    value = 0;

    for u = 1:size(R, 1)
        total = 0;

        for n = 1:size(R, 2)
            total = total + real(trace(R{u, n}));
        end

        value = value + w(u) * total;
    end

end

function data = attach_gradients(ctx, data, R, sortedRanges, order, delta)

    for n = 1:numel(data.C)
        data.gradient(:, n) = covariance_gradient(data.C{n}, ...
            data.Ablocks{n}, sortedRanges, order, delta, ctx.cb, ctx.w);
    end

    data.energy = weighted_energy(R, ctx.w);
    data.lagrangian = data.objective - data.energy;
end

function gradients = covariance_gradient(C, Ablocks, sortedRanges, order, delta, cb, w)
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
        gradient = (A' \ (Q / A)) / (cb * log(2)) - w(u) * eye(Lj);
        gradients{u} = hermitian(gradient);
    end

end

function [gains, calls] = probe_gains(pilot, X, ranges, Ltot, N, ridge, errorFloor)
    U = numel(ranges);
    Ly = pilot.numRx;
    count = size(X, 2);
    gains = zeros(U, N);
    level = 1;
    Atx = sqrt(level) * eye(Ltot);

    for n = 1:N
        Y = pilot.sample(n, Atx, X);

        if ~isnumeric(Y) || ~isequal(size(Y), [Ly, count]) || any(~isfinite(Y(:)))
            error('pilot.sample returned an invalid received-sample matrix');
        end

        Ryy = (Y * Y') / count;
        scale = max(real(trace(Ryy)) / max(Ly, 1), 1);
        Wle = (X * Y') / count / (hermitian(Ryy) + ridge * scale * eye(Ly));
        residual = X - Wle * Y;
        E = hermitian((residual * residual') / count);
        [Ve, De] = eig(E);
        e = min(max(real(diag(De)), errorFloor), 1);
        E = hermitian(Ve * diag(e) * Ve');
        J = hermitian(E \ eye(Ltot));
        excess = J - eye(Ltot);

        for u = 1:U
            idx = ranges{u};
            block = hermitian(excess(idx, idx));
            gains(u, n) = max(max(real(eig(block))), 0) / level;
        end

    end

    gains = max(gains, 1e-12);
    calls = N;
end

function [theta, ceiling] = initial_prices(gains, bu_min, w, cb)
    U = numel(bu_min);
    ceiling = zeros(U, 1);

    for u = 1:U
        target = bu_min;
        target(u) = target(u) + 1;
        weighted = 0;

        for v = 1:U
            weighted = weighted + w(v) * rate_waterfill(gains(v, :), target(v), cb);
        end

        ceiling(u) = weighted;
    end

    theta = max(ceiling(:), 1e-6) / 2;
end

function scale = price_energy_scale(theta, w, cb, gains)
    reference = max(theta(:) ./ w(:)) / (cb * log(2));
    fallback = 1 / max(mean(gains(:)), 1e-12);
    scale = max([real(reference), real(fallback), 1e-12]);
end

function total = rate_waterfill(gains, targetBits, cb)

    if targetBits <= 0
        total = 0;
        return;
    end

    sorted = sort(reshape(gains, 1, []), 'descend');
    N = numel(sorted);
    total = 0;

    for count = 1:N
        active = sorted(1:count);
        level = 2 ^ ((cb * targetBits - sum(log2(active))) / count);
        energies = level - 1 ./ active;

        if all(energies >= 0)
            total = sum(energies);

            if count == N || level <= 1 / sorted(min(count + 1, N))
                return;
            end

        end

    end

    if total == 0
        total = (2 ^ (cb * targetBits) - 1) / max(sorted(1), 1e-12);
    end

end

function [clusters, unique_values] = cluster_prices(theta, tol)
    values = double(theta(:));
    [unique_values, ~, groups] = uniquetol(values, tol);
    clusters = cell(numel(unique_values), 1);

    for k = 1:numel(unique_values)
        clusters{k} = find(groups == k)';
    end

end

function orders = decoding_orders(clusters, maxOrders)
    orders = perms(clusters{1});

    for k = 2:numel(clusters)
        extra = perms(clusters{k});
        grown = zeros(size(orders, 1) * size(extra, 1), ...
            size(orders, 2) + size(extra, 2));
        row = 0;

        for i = 1:size(orders, 1)

            for j = 1:size(extra, 1)
                row = row + 1;
                grown(row, :) = [orders(i, :), extra(j, :)];
            end

        end

        orders = grown;
    end

    if size(orders, 1) > maxOrders
        error('ada_minPMACMIMO:tooManyOrders', ...
            '%d tied decoding orders exceed maxOrders=%d', size(orders, 1), maxOrders);
    end

end

function [vertices, perTone] = measured_vertices(Jnat, orders, ranges, Lxu, cb)
    U = numel(ranges);
    N = numel(Jnat);
    K = size(orders, 1);
    vertices = zeros(U, K);
    perTone = cell(K, 1);

    for k = 1:K
        stacking = fliplr(orders(k, :));
        rows = concatenate_ranges(ranges(stacking));
        stops = cumsum(Lxu(stacking));
        starts = [1, stops(1:end - 1) + 1];
        rates = zeros(U, N);

        for n = 1:N
            block = hermitian(Jnat{n}(rows, rows));
            [C, flag] = chol(block);

            if flag ~= 0
                [V, D] = eig(block);
                block = hermitian(V * diag(max(real(diag(D)), realmin)) * V');
                C = chol(block);
            end

            S0 = real(diag(C)) .^ 2;

            for position = 1:U
                span = starts(position):stops(position);
                rates(stacking(position), n) = ...
                    sum(log(max(S0(span), realmin))) / (cb * log(2));
            end

        end

        perTone{k} = rates;
        vertices(:, k) = sum(rates, 2);
    end

end

function [pilot, Lxu, bu_min, w, cb, uniformFlag] = validate_inputs( ...
        pilot, Lxu, bu_min, w, cb)
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

    bu_min = bu_min(:);

    if isempty(bu_min) || any(~isfinite(bu_min)) || any(bu_min < 0)
        error('bu_min must contain finite nonnegative target rates');
    end

    w = w(:);

    if numel(w) ~= numel(bu_min) || any(~isfinite(w)) || any(w <= 0)
        error('w must contain one positive energy weight per user');
    end

    uniformFlag = isscalar(Lxu);

    if uniformFlag
        Lxu = repmat(Lxu, 1, numel(bu_min));
    else
        Lxu = reshape(Lxu, 1, []);
    end

    if numel(Lxu) ~= numel(bu_min) || any(Lxu < 1) || any(Lxu ~= fix(Lxu))
        error('Lxu must contain one positive integer per user');
    end

    if ~isscalar(cb) || ~ismember(cb, [1, 2])
        error('cb must be one or two');
    end

end

function validate_options(numPilots, maxCovariance, maxBacktrack, stepShrink, ...
        armijo, ridge, errorFloor, patience)
    integers = [numPilots, maxCovariance, maxBacktrack, patience];

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

function R = initial_covariance(Lxu, N, floorValue)
    U = numel(Lxu);
    R = cell(U, N);

    for u = 1:U

        for n = 1:N
            R{u, n} = floorValue * eye(Lxu(u));
        end

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

function [R, factors] = project_covariances(R, floorValue, ranges)
    U = size(R, 1);
    N = size(R, 2);

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
            [V, D] = eig(hermitian(R{u, n}));
            values = max(real(diag(D)), floorValue);
            R{u, n} = hermitian(V * diag(values) * V');

            if nargout > 1
                [factorValues, permutation] = sort(values, 'descend');
                A = V(:, permutation) * diag(sqrt(factorValues));
                factors.Ablocks{u, n} = A;
                factors.Atx{n}(ranges{u}, ranges{u}) = A;
            end

        end

    end

end

function R = remove_exploration_floor(R, floorValue)

    for i = 1:numel(R)
        [V, D] = eig(hermitian(R{i}));
        values = max(real(diag(D)) - floorValue, 0);
        R{i} = hermitian(V * diag(values) * V');
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
