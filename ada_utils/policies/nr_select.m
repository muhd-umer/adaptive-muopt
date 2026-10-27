function [actions, out] = nr_select(K, catalogs, current, spec, opts)
    %NR_SELECT Pick NR-realizable transmit actions from an effective Gram belief
    %
    % Each UE transmits one wideband codebook precoder at a flat power
    % spectral density, as PUSCH does. The selector runs coordinate ascent
    % over UEs on the weighted rate. That objective is concave in every R_u
    % for a fixed decoding order, so the linearization from the learned
    % covariance gradient upper-bounds every candidate; the candidates are
    % scored in bound order and the scan stops once no bound can beat the
    % incumbent, which keeps the search exact per coordinate
    %
    % Args:
    %   K: Ltot-by-Ltot-by-T effective Gram belief on the evaluation tones
    %   catalogs: 1-by-U cell of nr_catalog outputs
    %   current: 1-by-U struct array of deployed actions with row, powerDb
    %     (backoff from the cap, at most zero), and active
    %   spec: Objective with theta (weights), cb, Lxu, pcap (per-RE energy cap
    %     of each UE), and optional etotal (shared per-RE energy budget) and
    %     toneWeights
    %   opts: Optional reachable ('any' for every grid level, 'tpc' for one
    %     accumulated TPC step), powerGridDb (0:-1:-20), tpcStepsDb
    %     ([-1 0 1 3]), maxSweeps (4), screen (true), and multiStart (false)
    %
    % Returns:
    %   actions: 1-by-U struct array with row, powerDb, tpc code point (NaN
    %     when no single TPC command reaches it), and changed
    %   out: Selected and current objective values, gain, per-user rates and
    %     energies, and evaluation counts

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    U = numel(catalogs);
    T = size(K, 3);
    spec.theta = spec.theta(:);
    spec.pcap = spec.pcap(:);
    spec.etotal = get_opt(spec, 'etotal', inf);
    spec.toneWeights = get_opt(spec, 'toneWeights', ones(1, T) / T);
    grid = get_opt(opts, 'powerGridDb', 0:-1:-20);
    reachable = get_opt(opts, 'reachable', 'any');
    steps = get_opt(opts, 'tpcStepsDb', [-1 0 1 3]);
    maxSweeps = get_opt(opts, 'maxSweeps', 4);
    screen = get_opt(opts, 'screen', true);
    multiStart = get_opt(opts, 'multiStart', false);
    minDb = min(grid);
    active = [current.active];

    counters = struct('evaluations', 0, 'screened', 0);
    candidates = cell(1, U);

    for u = 1:U

        if ~active(u)
            candidates{u} = zeros(0, 2);
            continue;
        end

        if strcmp(reachable, 'tpc')
            levels = unique(min(max(current(u).powerDb + steps, minDb), 0));
        else
            levels = grid;
        end

        [rowGrid, levelGrid] = ndgrid(1:numel(catalogs{u}.rows), levels);
        candidates{u} = [rowGrid(:), levelGrid(:)];
    end

    start = struct('rows', [current.row], 'pdb', [current.powerDb]);
    start.pdb(~active) = -inf;
    [currentValue, currentEval] = evaluate(K, catalogs, start, spec, active);

    % a start that breaks the shared budget is backed off uniformly so the
    % search begins from a feasible configuration
    if ~currentEval.feasible
        excess = 10 * log10(sum(currentEval.energy) / spec.etotal);
        start.pdb(active) = max(start.pdb(active) - ceil(excess - 1e-12), minDb);
        [currentValue, currentEval] = evaluate(K, catalogs, start, spec, active);
    end

    counters.evaluations = counters.evaluations + 1;
    starts = {start};

    if multiStart
        starts = [starts, extra_starts(K, catalogs, start, spec, active, minDb)];
    end

    bestState = start;
    bestValue = currentValue;
    bestEval = currentEval;

    for s = 1:numel(starts)
        [state, value, eval, counters] = coordinate_search(K, catalogs, starts{s}, ...
            spec, active, candidates, maxSweeps, screen, counters);

        if value > bestValue + 1e-12 * max(1, abs(bestValue))
            bestValue = value;
            bestState = state;
            bestEval = eval;
        end

    end

    actions = struct('row', num2cell(bestState.rows), 'powerDb', num2cell(bestState.pdb), ...
        'tpc', num2cell(nan(1, U)), 'changed', num2cell(false(1, U)));

    for u = find(active)
        actions(u).tpc = tpc_code(current(u).powerDb, bestState.pdb(u), steps, minDb);
        actions(u).changed = bestState.rows(u) ~= current(u).row || ...
            bestState.pdb(u) ~= current(u).powerDb;
    end

    for u = find(~active)
        actions(u).powerDb = current(u).powerDb;
    end

    out = struct();
    out.value = bestValue;
    out.currentValue = currentValue;
    out.gain = bestValue - currentValue;
    out.objective = bestEval.objective;
    out.rates = bestEval.rates;
    out.energy = bestEval.energy;
    out.evaluations = counters.evaluations;
    out.screened = counters.screened;
end

function [state, value, eval, counters] = coordinate_search(K, catalogs, state, spec, ...
        active, candidates, maxSweeps, screen, counters)
    [value, eval] = evaluate(K, catalogs, state, spec, active);
    counters.evaluations = counters.evaluations + 1;
    users = find(active);

    for sweep = 1:maxSweeps
        changed = false;

        for u = users
            [state, value, eval, counters, moved] = coordinate_step(K, catalogs, ...
                state, value, eval, spec, active, u, candidates{u}, screen, counters);
            changed = changed || moved;
        end

        if ~changed
            break;
        end

    end

end

function [state, value, eval, counters, moved] = coordinate_step(K, catalogs, state, ...
        value, eval, spec, active, u, list, screen, counters)
    moved = false;

    if isempty(list)
        return;
    end

    catalog = catalogs{u};
    energies = spec.pcap(u) * 10 .^ (list(:, 2) / 10) .* reshape(catalog.eta(list(:, 1)), [], 1);
    others = sum(eval.energy) - eval.energy(u);
    list = list(others + energies <= spec.etotal * (1 + 1e-9), :);

    if isempty(list)
        return;
    end

    bounds = inf(size(list, 1), 1);

    if screen
        [~, gradientEval] = evaluate(K, catalogs, state, spec, active, true);
        counters.evaluations = counters.evaluations + 1;
        G = gradientEval.meanGradient{u};
        Ru = gradientEval.covariances{u};
        linear = zeros(numel(catalog.rows), 1);

        for r = 1:numel(catalog.rows)
            F = catalog.F{r};
            linear(r) = real(trace(F' * G * F));
        end

        bounds = value + spec.pcap(u) * 10 .^ (list(:, 2) / 10) .* linear(list(:, 1)) - ...
            real(trace(G * Ru));
    end

    [~, sequence] = sort(bounds, 'descend');
    tolerance = 1e-12 * max(1, abs(value));

    for idx = sequence(:)'

        if bounds(idx) <= value + tolerance
            counters.screened = counters.screened + 1;
            continue;
        end

        trial = state;
        trial.rows(u) = list(idx, 1);
        trial.pdb(u) = list(idx, 2);

        if trial.rows(u) == state.rows(u) && trial.pdb(u) == state.pdb(u)
            continue;
        end

        [trialValue, trialEval] = evaluate(K, catalogs, trial, spec, active);
        counters.evaluations = counters.evaluations + 1;

        if trialValue > value + tolerance
            state = trial;
            value = trialValue;
            eval = trialEval;
            moved = true;
        end

    end

end

function [value, eval] = evaluate(K, catalogs, state, spec, active, wantGradient)

    if nargin < 6
        wantGradient = false;
    end

    U = numel(catalogs);
    F = cell(1, U);
    energy = zeros(U, 1);

    for u = 1:U

        if ~active(u)
            F{u} = zeros(catalogs{u}.numPorts, 0);
            continue;
        end

        level = spec.pcap(u) * 10 ^ (state.pdb(u) / 10);
        F{u} = sqrt(level) * catalogs{u}.F{state.rows(u)};
        energy(u) = level * catalogs{u}.eta(state.rows(u));
    end

    result = mac_eval(K, F, spec.Lxu, spec.theta, spec.cb, ...
        struct('toneWeights', spec.toneWeights, 'gradient', wantGradient));
    eval = struct('energy', energy, 'objective', result.objective, ...
        'rates', result.userRates, 'feasible', sum(energy) <= spec.etotal * (1 + 1e-9));
    value = result.objective;

    if ~eval.feasible
        value = -inf;
    end

    if wantGradient
        eval.meanGradient = cell(1, U);
        eval.covariances = cell(1, U);
        weights = reshape(spec.toneWeights, 1, 1, []);

        for u = 1:U
            eval.meanGradient{u} = sum(result.gradient{u} .* weights, 3);
            eval.covariances{u} = F{u} * F{u}';
        end

    end

end

function starts = extra_starts(K, catalogs, start, spec, active, minDb)
    %EXTRA_STARTS Full-power restarts that keep coordinate search out of poor optima
    full = start;
    full.pdb(active) = 0;
    starts = {full};
    rateSpec = spec;
    rateSpec.etotal = inf;
    best = full;

    for u = find(active)
        bestValue = -inf;

        for r = 1:numel(catalogs{u}.rows)
            trial = best;
            trial.rows(u) = r;
            value = evaluate(K, catalogs, trial, rateSpec, active);

            if value > bestValue
                bestValue = value;
                best.rows(u) = r;
            end

        end

    end

    starts{end + 1} = best;

    if isfinite(spec.etotal)

        for i = 1:numel(starts)
            used = 0;

            for u = find(active)
                used = used + spec.pcap(u) * catalogs{u}.eta(starts{i}.rows(u));
            end

            backoff = max(0, ceil(10 * log10(used / spec.etotal) - 1e-12));
            starts{i}.pdb(active) = max(-backoff, minDb);
        end

    end

end

function code = tpc_code(fromDb, toDb, steps, minDb)
    code = nan;
    reached = min(max(fromDb + steps, minDb), 0);
    matches = find(abs(reached - toDb) < 1e-9);

    if isempty(matches)
        return;
    end

    [~, pick] = min(abs(steps(matches)));
    code = matches(pick) - 1;
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
