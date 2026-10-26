function method = method_covariance()
    %METHOD_COVARIANCE Compressed covariance feedback driven by the adaptive solvers
    %
    % Every UE transmits a full-rank per-band covariance, the gNB learns from
    % its DMRS, runs the matching adaptive solver, and sends a quantized
    % covariance message (nr_cov_policy), a nonstandard payload
    %
    % Returns:
    %   method: Handles init(name, ctx, active) and step(state, ctx, input,
    %     record), and the method kind

    method = struct('kind', 'covariance', 'init', @init, 'step', @step);
end

function state = init(name, ctx, active)
    env = ctx.env;
    modeB = ctx.modeB;
    state = method_state(name, ctx, active);
    state.numTones = ceil(env.numGroups / modeB.bundle);
    state.tones = arrayfun(@(k) (k - 1) * modeB.bundle + 1:min(k * modeB.bundle, ...
        env.numGroups), 1:state.numTones, 'UniformOutput', false);
    state.toneOfGroup = ceil((1:env.numGroups) / modeB.bundle);
    state.floor = modeB.floorFraction * env.pcap(:) ./ env.Lxu(:);
    state.messages = arrayfun(@(L) cov_message(L, state.numTones, modeB), env.Lxu, ...
        'UniformOutput', false);
    [state.R, state.params] = default_covariances(ctx, state, active);
    state.trace = zeros(0, 4);
    state.silentRows = cell2mat(env.ranges(~active));
end

function [state, record] = step(state, ctx, input, record)
    env = ctx.env;
    H = input.H;
    [state, record] = deliver(state, ctx, input.active, input.slot, record);
    factors = cov_factors(state.R);
    groupIndex = ceil((1:env.numSubcarriers)' / (env.numSubcarriers / env.numGroups));
    F = cell(1, env.U);

    for u = 1:env.U
        F{u} = cat(3, factors{u, state.toneOfGroup(groupIndex)});
    end

    result = mac_eval(input.truth.Keval, F, env.Lxu, ctx.objective.theta, ctx.cfg.cb);
    record.rates = result.userRates;
    record.energy = cellfun(@(R) real(trace(R)), state.R) * ...
        (cellfun(@numel, state.tones(:)) / env.numGroups);
    record.objective = result.objective;
    [state, record] = link_outcome(state, ctx, record, input.t);

    store = containers.Map('KeyType', 'double', 'ValueType', 'any');
    pilot = struct('numRx', size(H, 1), 'numTones', state.numTones, ...
        'observe', @(g, Atx) observe(H, env, state.tones{g}, g, Atx, store));
    [state, sends] = nr_cov_policy(state, ctx, pilot, store, input.slot);

    for i = 1:numel(sends)
        u = sends(i).ue;
        payload = struct('bits', {sends(i).payload}, 'R', {sends(i).R}, ...
            'reference', {state.params{u}});
        state.ctrl = ctrl_send(state.ctrl, input.slot, u, 'covariance', sends(i).bits, payload);
        state.pendingUntil(u) = state.ctrl.queue(end).due;
        record.designBits = record.designBits + sends(i).bits;
        record.dciBits = record.dciBits + sends(i).bits + env.dci.crc;
        record.messages = record.messages + 1;
    end

    % the policy skips its measurement while a message is in flight, yet the
    % gNB still receives this slot's DMRS and plans the next MCS from it
    if store.Count < state.numTones
        factors = cov_factors(state.R);

        for g = 1:state.numTones
            blocks = factors(:, g);
            observe(H, env, state.tones{g}, g, blkdiag(blocks{:}), store);
        end

    end

    record.dataFraction = data_fraction(env, false);
    state = plan(state, ctx, store, input.slot);
end

function [state, record] = deliver(state, ctx, active, slot, record)
    %DELIVER Apply the covariance messages that land in this slot
    %
    % The UE decodes the payload against the reference both ends hold and
    % projects it onto its floor and budget
    env = ctx.env;
    G = state.numTones;
    [state.ctrl, delivered, lost] = ctrl_deliver(state.ctrl, slot);
    record.lost = record.lost + numel(lost);
    budgets = env.pcap(:) * G .* state.active(:);

    for i = 1:numel(delivered)
        u = delivered(i).ue;
        [state.R(u, :), state.params{u}] = state.messages{u}.decode( ...
            delivered(i).payload.bits, budgets(u), delivered(i).payload.reference);
    end

    if ~isequal(active, state.active)
        rejoined = active & ~state.active;
        state.active = active;
        state.silentRows = cell2mat(env.ranges(~active));

        if any(rejoined)
            [fresh, freshParams] = default_covariances(ctx, state, active);
            state.R(rejoined, :) = fresh(rejoined, :);
            state.params(rejoined) = freshParams(rejoined);
        end

    end

    state.R = cov_project(state.R, env.pcap(:) * G .* active(:), state.floor(:) .* active(:));
    [state.R, refit] = fit_shared(state.R, ctx.selection.etotal, active);

    if refit
        record.messages = record.messages + nnz(active);
    end

end

function [R, params] = default_covariances(ctx, state, active)
    %DEFAULT_COVARIANCES Equal-power identity design both ends start from
    env = ctx.env;
    G = state.numTones;
    R = cell(env.U, G);
    params = cell(1, env.U);
    energy = min(env.pcap(:) .* active(:), ctx.selection.etotal / max(nnz(active), 1));

    for u = 1:env.U
        R(u, :) = {energy(u) / env.Lxu(u) * eye(env.Lxu(u))};
    end

    budgets = env.pcap(:) * G .* active(:);
    R = cov_project(R, budgets, state.floor(:) .* active(:));

    for u = find(budgets(:)' > 0)
        bits = state.messages{u}.encode(R(u, :), budgets(u), []);
        [~, params{u}] = state.messages{u}.decode(bits, budgets(u), []);
    end

end

function [Y, X] = observe(H, env, groups, tone, Atx, store)
    %OBSERVE DMRS of the full-rank covariance design on the PRGs of one tone
    positions = env.dmrs.positions;
    members = find(ismember(env.dmrs.groupOf, groups));
    X = zeros(env.Ltot, numel(members));

    for u = 1:env.U
        values = env.dmrs.values{u, env.Lxu(u)};
        X(env.ranges{u}, :) = transpose(values(members, :));
    end

    linear = sub2ind([env.numSubcarriers, env.numSymbols], positions(members, 1), ...
        positions(members, 2));
    flat = reshape(H, size(H, 1), size(H, 2), []);
    S = Atx * X;
    Y = reshape(pagemtimes(flat(:, :, linear), reshape(S, size(S, 1), 1, [])), size(H, 1), []);
    Y = Y + env.noiseFactor * (randn(size(Y)) + 1j * randn(size(Y))) / sqrt(2);
    store(tone) = struct('Y', Y, 'X', X, 'Atx', Atx);
end

function state = plan(state, ctx, store, slot)
    %PLAN MCS for the uplink slot the current decision first reaches
    env = ctx.env;
    G = state.numTones;
    target = next_occasion(ctx, slot + ctx.cfg.control.delay);

    if target > size(state.plan, 2) || store.Count < G
        return;
    end

    K = zeros(env.Ltot, env.Ltot, G);

    for g = 1:G
        entry = store(g);
        J = mmse_info(entry.Y, entry.X, struct('silentRows', state.silentRows));
        K(:, :, g) = gram_fuse([], J, entry.Atx, 1);
    end

    pending = state.R;

    for message = state.ctrl.queue

        if message.due <= target
            pending(message.ue, :) = message.payload.R;
        end

    end

    factors = cov_factors(pending);
    F = cell(1, env.U);

    for u = 1:env.U
        F{u} = cat(3, factors{u, :});
    end

    result = mac_eval(K, F, env.Lxu, ctx.objective.theta, ctx.cfg.cb, ...
        struct('toneWeights', cellfun(@numel, state.tones) / env.numGroups));
    state.plan(:, target) = nr_mcs(max(result.userRates - state.backoff, 0), ...
        env.Lxu(:) .* state.active(:));
end
