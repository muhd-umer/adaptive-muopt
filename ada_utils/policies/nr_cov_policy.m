function [state, sends] = nr_cov_policy(state, ctx, pilot, store, slot)
    %NR_COV_POLICY Compressed continuous-covariance adaptation, a nonstandard extension
    %
    % Every UE transmits a full-rank covariance per frequency band whose weak
    % directions sit at an exploration floor, so its DMRS reveals the whole
    % MMSE information. One single-pass step of the matching adaptive solver
    % turns that measurement into a proposed design, which is quantized by the
    % per-UE covariance message, projected by the rule the UE applies, and
    % signaled only when the learned information predicts a gain for the
    % decoded design rather than the unquantized one. The payload is custom
    % signaling, so this mode is compared with the NR-native mode at matched
    % control bits rather than claimed as an NR capability
    %
    % Args:
    %   state: Policy state with the deployed design, prices, and references
    %   ctx: Shared run context with env, objective, and Mode B settings
    %   pilot: Observe-style pilot interface on this slot's DMRS, one tone per band
    %   store: Map from band to the (Y, X, Atx) the interface produced
    %   slot: Current slot
    %
    % Returns:
    %   state: Updated prices, trace, and evaluation count
    %   sends: Struct array of UE, payload bits, message bits, decoded design,
    %     and the codec parameters both ends keep as the differential reference

    env = ctx.env;
    modeB = ctx.modeB;
    spec = ctx.objective;
    G = state.numTones;
    sends = struct('ue', {}, 'bits', {}, 'payload', {}, 'R', {}, 'params', {});

    if any(state.pendingUntil > slot)
        return;
    end

    active = state.active;
    caps = env.pcap(:) * G .* active(:);
    floors = state.floor(:) .* active(:);
    step = modeB.stepFraction * norm(cellfun(@(R) norm(R, 'fro'), state.R(:)));

    [R, info] = solver_steps(state, ctx, pilot, store, caps, floors, step);
    state.evaluations = state.evaluations + info.pilotCalls;
    K = learned_gram(store, state, G);
    [decoded, params, payloads, bits] = quantize(state, R, ctx, caps);
    decoded = cov_project(decoded, caps, floors);

    if strcmp(spec.type, 'maxresmac')
        [decoded, params, payloads, bits] = enforce_shared_budget(state, decoded, ...
            params, payloads, bits, ctx, ctx.selection.etotal * G, caps, floors);
    end

    current = design_value(K, state.R, spec, state, env);
    proposed = design_value(K, decoded, spec, state, env);
    unquantized = design_value(K, cov_project(R, caps, floors), spec, state, env);
    state.trace(end + 1, :) = [slot, current, unquantized, proposed];

    if proposed - current <= modeB.triggerRelative * max(abs(current), 1e-9)
        return;
    end

    for u = find(active(:)')
        sends(end + 1) = struct('ue', u, 'bits', bits(u), 'payload', {payloads{u}}, ...
            'R', {decoded(u, :)}, 'params', {params{u}}); %#ok<AGROW>
    end

end

function K = learned_gram(store, state, G)
    K = [];

    for g = 1:G
        record = store(g);
        J = mmse_info(record.Y, record.X, struct('silentRows', state.silentRows));
        Kg = gram_fuse([], J, record.Atx, 1);

        if isempty(K)
            K = zeros(size(Kg, 1), size(Kg, 2), G);
        end

        K(:, :, g) = Kg;
    end

end

function [decoded, params, payloads, bits] = quantize(state, R, ctx, caps)
    env = ctx.env;
    U = env.U;
    G = state.numTones;
    decoded = cell(U, G);
    params = cell(1, U);
    payloads = cell(1, U);
    bits = zeros(U, 1);

    for u = 1:U

        if ~state.active(u)
            params{u} = state.params{u};

            for g = 1:G
                decoded{u, g} = zeros(env.Lxu(u));
            end

            continue;
        end

        message = state.messages{u};
        [payloads{u}, ~] = message.encode(R(u, :), caps(u), state.params{u});
        [decoded(u, :), params{u}] = message.decode(payloads{u}, caps(u), state.params{u});

        bits(u) = numel(payloads{u});
    end

end

function [decoded, params, payloads, bits] = enforce_shared_budget(state, decoded, ...
        params, payloads, bits, ctx, total, caps, floors)
    %ENFORCE_SHARED_BUDGET Back the decoded designs off until the shared budget holds
    energy = sum(cellfun(@(R) real(trace(R)), decoded(:)));

    for attempt = 1:12

        if energy <= total * (1 + 1e-9)
            return;
        end

        scaled = decoded;

        for i = 1:numel(scaled)
            scaled{i} = scaled{i} * (total / energy) * 0.97 ^ attempt;
        end

        [decoded, params, payloads, bits] = quantize(state, scaled, ctx, caps);
        decoded = cov_project(decoded, caps, floors);
        energy = sum(cellfun(@(R) real(trace(R)), decoded(:)));
    end

end

function value = design_value(K, R, spec, state, env)
    U = env.U;
    G = state.numTones;
    F = cell(1, U);

    for u = 1:U
        F{u} = zeros(env.Lxu(u), env.Lxu(u), G);

        for g = 1:G
            [V, D] = eig(0.5 * (R{u, g} + R{u, g}'));
            F{u}(:, :, g) = V * diag(sqrt(max(real(diag(D)), 0)));
        end

    end

    result = mac_eval(K, F, env.Lxu, spec.theta, 1);
    value = result.objective;

end

function cells = tensor_cells(R, Lxu)

    if iscell(R)
        cells = R;
        return;
    end

    U = size(R, 3);
    G = size(R, 4);
    cells = cell(U, G);

    for u = 1:U

        for g = 1:G
            cells{u, g} = R(1:Lxu(u), 1:Lxu(u), u, g);
        end

    end

end

function [R, info] = solver_steps(state, ctx, pilot, store, caps, floors, step)
    %SOLVER_STEPS Matching-solver iterations on the information this slot's DMRS revealed
    %
    % The first step measures the real pilots; the remaining innerSteps - 1
    % replay that measurement through a virtual sampler built from the learned
    % Gram square root, so they cost no pilots and form no channel matrix
    env = ctx.env;
    G = state.numTones;
    opts = struct('singlePass', true, 'R0', {state.R}, 'floorInitial', floors, ...
        'floorFinal', floors / 10, 'stepInitial', step, 'numPilots', env.Ltot);
    [R, info] = run_solver(pilot, ctx, caps, opts);
    calls = info.pilotCalls;

    if ctx.modeB.innerSteps > 1
        K = learned_gram(store, state, G);
        roots = cell(1, G);

        for g = 1:G
            [V, D] = eig(0.5 * (K(:, :, g) + K(:, :, g)'));
            roots{g} = diag(sqrt(max(real(diag(D)), 0))) * V';
        end

        virtual = struct('numRx', env.Ltot, 'numTones', G, ...
            'sample', @(n, Atx, X) roots{n} * Atx * X + exact_noise(env.Ltot, X));
        opts = struct('R0', {tensor_cells(R, env.Lxu)}, 'floorInitial', floors, ...
            'floorFinal', floors / 10, 'maxOuter', ctx.modeB.innerSteps - 1, ...
            'numPilots', 4 * env.Ltot);
        R = run_solver(virtual, ctx, caps, opts);
    end

    R = tensor_cells(R, env.Lxu);
    info.pilotCalls = calls;
end

function [R, info] = run_solver(pilot, ctx, caps, opts)
    spec = ctx.objective;
    G = pilot.numTones;

    switch spec.type
        case 'maxresmac'
            opts.Pmax = caps;
            [R, ~, ~, ~, info] = ada_maxRESMACMIMO(pilot, ctx.env.Lxu, ...
                ctx.selection.etotal * G, spec.theta, 1, opts);
        case 'maxrmac'
            [R, ~, ~, ~, info] = ada_maxRMACMIMO(pilot, ctx.env.Lxu, caps, spec.theta, 1, opts);
        otherwise
            error('unknown objective %s', spec.type);
    end

end

function Z = exact_noise(rows, X)
    %EXACT_NOISE Unit-covariance noise orthogonal to the pilots, so replayed moments are exact
    count = size(X, 2);
    Z = exp(2j * pi * (1:rows)' * (0:count - 1) / count + 0.37);
    Z = Z - (Z * X') / (X * X') * X;
    covariance = (Z * Z') / count;
    Z = chol(0.5 * (covariance + covariance'), 'lower') \ Z;
end
