function method = method_codebook()
    %METHOD_CODEBOOK Codebook-based PUSCH: wideband TPMI and rank with accumulated TPC
    %
    % 'nr-csi' and 'nr-pilot' run the gNB scheduler of nr_codebook_policy on
    % the SRS and DMRS they observe and signal bit-exact DCI 0_1 fields.
    % 'nr-oracle' picks the best codebook action on the true channel with no
    % delay, and 'hold' keeps the full-rank identity precoder at full power
    % with an MCS from the true rate; both are genie references
    %
    % Returns:
    %   method: Handles init(name, ctx, active) and step(state, ctx, input,
    %     record), and the method kind

    method = struct('kind', 'codebook', 'init', @init, 'step', @step);
end

function state = init(name, ctx, active)
    env = ctx.env;
    state = method_state(name, ctx, active);
    state.deployed = struct('row', num2cell(env.defaultRow), ...
        'powerDb', num2cell(zeros(1, env.U)), 'active', num2cell(active));
    total = sum(env.pcap(active));

    if total > ctx.selection.etotal
        backoff = -ceil(10 * log10(total / ctx.selection.etotal) - 1e-12);
        [state.deployed.powerDb] = deal(backoff);
    end

    state.planned = state.deployed;
    state.K = zeros(env.Ltot, env.Ltot, env.numGroups);
    state.ready = false;
end

function [state, record] = step(state, ctx, input, record)

    switch state.name
        case 'hold'
            state.deployed = set_active(state.deployed, input.active);
            record = evaluate(state, ctx, input, record);
            record = genie_link(state, ctx, record);
        case 'nr-oracle'
            state.deployed = set_active(state.deployed, input.active);
            spec = ctx.selection;
            spec.toneWeights = ones(1, size(input.truth.Ksel, 3)) / size(input.truth.Ksel, 3);
            opts = ctx.selectOpts;
            opts.reachable = 'any';
            opts.multiStart = true;
            [actions, out] = nr_select(input.truth.Ksel, ctx.env.catalogs, state.deployed, ...
                spec, opts);
            state.evaluations = state.evaluations + out.evaluations;
            [state.deployed.row] = actions.row;
            [state.deployed.powerDb] = actions.powerDb;
            record = evaluate(state, ctx, input, record);
            record = genie_link(state, ctx, record);
        case {'nr-csi', 'nr-pilot'}
            [state, record] = scheduled(state, ctx, input, record);
    end

end

function [state, record] = scheduled(state, ctx, input, record)
    env = ctx.env;
    slot = input.slot;
    [state, record] = deliver(state, ctx, input.active, slot, record);
    record = evaluate(state, ctx, input, record);
    [state, record] = link_outcome(state, ctx, record, input.t);
    dmrs = nr_dmrs_observe(input.H, env, factors(ctx, state.deployed));
    srs = [];

    if ~isnan(input.sounding.slot) && any(input.active)
        srs = nr_srs_observe(input.sounding.H, env, input.active);
    end

    [state, sends] = nr_codebook_policy(state, ctx, srs, dmrs, erase(state.name, 'nr-'));

    for i = 1:numel(sends)
        u = sends(i).ue;
        bits = env.dci.encode(u, sends(i).row, sends(i).tpc);
        state.ctrl = ctrl_send(state.ctrl, slot, u, 'dci01', numel(bits), bits);
        state.pendingUntil(u) = state.ctrl.queue(end).due;
        state.planned(u).row = sends(i).row;
        state.planned(u).powerDb = sends(i).powerDb;
        record.designBits = record.designBits + numel(bits);
        record.dciBits = record.dciBits + env.dci.dci01(u);
        record.messages = record.messages + 1;
    end

    record.dataFraction = data_fraction(env, input.sounding.slot == slot);
    target = next_occasion(ctx, slot + ctx.cfg.control.delay);

    if state.ready && target <= size(state.plan, 2)
        F = factors(ctx, state.planned);
        result = mac_eval(state.K, F, env.Lxu, ctx.objective.theta, ctx.cfg.cb);
        state.plan(:, target) = nr_mcs(max(result.userRates - state.backoff, 0), ...
            cellfun(@(f) size(f, 2), F)');
    end

end

function [state, record] = deliver(state, ctx, active, slot, record)
    env = ctx.env;
    floorDb = ctx.cfg.power.floorDb;
    [state.ctrl, delivered, lost] = ctrl_deliver(state.ctrl, slot);

    for i = 1:numel(delivered)
        u = delivered(i).ue;
        msg = env.dci.decode(u, delivered(i).payload);

        if msg.valid
            state.deployed(u).row = msg.row;
            state.deployed(u).powerDb = min(max(state.deployed(u).powerDb + msg.tpcDb, ...
                floorDb), 0);
        end

    end

    % a lost grant is detected when the UE fails to use it, and the scheduler
    % falls back to what the UE really runs
    for i = 1:numel(lost)
        state.planned(lost(i).ue) = state.deployed(lost(i).ue);
        record.lost = record.lost + 1;
    end

    for u = find(state.pendingUntil <= slot)
        state.planned(u).row = state.deployed(u).row;
        state.planned(u).powerDb = state.deployed(u).powerDb;
    end

    state.deployed = set_active(state.deployed, active);
    state.planned = set_active(state.planned, active);

    % a scheduling change gives every UE a new grant, so the gNB backs the
    % scheduled UEs off together when a returning UE would break the budget
    if ~isequal(active, state.active)
        state.active = active;
        energy = sum(action_energy(ctx, state.planned));

        if energy > ctx.selection.etotal * (1 + 1e-9)
            backoff = ceil(10 * log10(energy / ctx.selection.etotal) - 1e-12);

            for u = find(active)
                state.planned(u).powerDb = max(state.planned(u).powerDb - backoff, floorDb);
                state.deployed(u).powerDb = max(state.deployed(u).powerDb - backoff, floorDb);
            end

            record.designBits = record.designBits + sum(env.dci.design(active));
            record.dciBits = record.dciBits + sum(env.dci.dci01(active));
            record.messages = record.messages + nnz(active);
        end

    end

end

function record = evaluate(state, ctx, input, record)
    env = ctx.env;
    result = mac_eval(input.truth.Keval, factors(ctx, state.deployed), env.Lxu, ...
        ctx.objective.theta, ctx.cfg.cb);
    record.rates = result.userRates;
    record.energy = action_energy(ctx, state.deployed);
    record.objective = result.objective;
    record.deployedRow = [state.deployed.row]';
    record.deployedPower = [state.deployed.powerDb]';
    record.dataFraction = data_fraction(env, false);
end

function record = genie_link(state, ctx, record)
    ranks = cellfun(@(f) size(f, 2), factors(ctx, state.deployed))';
    record.throughput = nr_mcs(record.rates, ranks);
end

function actions = set_active(actions, active)
    active = num2cell(active);
    [actions.active] = active{:};
end

function F = factors(ctx, actions)
    env = ctx.env;
    F = cell(1, env.U);

    for u = 1:env.U

        if actions(u).active
            F{u} = sqrt(env.pcap(u) * 10 ^ (actions(u).powerDb / 10)) * ...
                env.catalogs{u}.F{actions(u).row};
        else
            F{u} = zeros(env.Lxu(u), 0);
        end

    end

end

function energy = action_energy(ctx, actions)
    env = ctx.env;
    energy = zeros(env.U, 1);

    for u = find([actions.active])
        energy(u) = env.pcap(u) * 10 ^ (actions(u).powerDb / 10) * ...
            env.catalogs{u}.eta(actions(u).row);
    end

end
