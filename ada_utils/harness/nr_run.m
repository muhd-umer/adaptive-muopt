function result = nr_run(cfg, env, spec, realization)
    %NR_RUN Simulate one channel realization of an NR uplink MAC experiment
    %
    % Every method sees the same channel, noise seed, pilot resources, UE
    % capabilities, and control link. The run steps through the uplink slots
    % of the TDD pattern while the channel evolves continuously in between.
    % In each uplink slot the control messages that have landed are applied,
    % the deployed design is scored on the true channel, its DMRS and the
    % latest SRS are observed, and only then does each policy decide, so a
    % decision taken in slot s first takes effect in the first uplink slot at
    % or after s + delay. The reciprocity methods sound their CSI-RS in the
    % first downlink or special slot at least lagSlots after the DMRS, and
    % the UE's new precoder waits processingSlots after the CSI-RS
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %   env: Environment from nr_setup
    %   spec: Objective with type ('maxresmac' or 'maxrmac'), theta, and an
    %     optional sumFraction of the summed caps for the shared budget
    %   realization: Realization index, which fixes every random stream
    %
    % Returns:
    %   result: Per-method rates, energies, objectives, link throughput,
    %     control bits and messages, and overhead in every uplink slot, and
    %     the slot numbers of those uplink slots

    rng(cfg.seed + realization);
    chan = nr_channels(cfg, env, cfg.seed + realization);
    U = env.U;
    activity = cfg.activity;

    if isempty(activity)
        activity = true(U, cfg.horizonSlots);
    end

    ctx = run_context(cfg, env, spec, realization);
    T = numel(ctx.slots);
    names = cfg.methods;
    M = numel(names);
    methods = cellfun(@method_for, names, 'UniformOutput', false);
    methods = [methods{:}];
    states = cell(1, M);
    first = activity(:, ctx.slots(1) + 1)';

    for m = 1:M
        states{m} = methods(m).init(names{m}, ctx, first);
    end

    needsCsirs = any(strcmp({methods.kind}, 'reciprocity'));
    needsSrs = any(strcmp({methods.kind}, 'codebook'));
    result = struct('names', {names}, 'spec', spec, 'realization', realization, ...
        'slots', ctx.slots);
    result.metrics = repmat({empty_metrics(U, T)}, 1, M);

    for t = 1:T
        slot = ctx.slots(t);
        [H, chan] = nr_channel_slot(chan, env, slot);
        input = struct('slot', slot, 't', t, 'H', H, 'truth', true_gram(H, env, cfg), ...
            'active', activity(:, slot + 1)');
        input.downlink = struct('slot', ctx.csirsSlots(t), 'H', H);
        input.sounding = struct('slot', ctx.srsSlots(t), 'H', H);

        if needsCsirs && input.downlink.slot ~= slot
            [input.downlink.H, chan] = nr_channel_slot(chan, env, input.downlink.slot);
        end

        if needsSrs && ~isnan(input.sounding.slot) && input.sounding.slot ~= slot
            [input.sounding.H, chan] = nr_channel_slot(chan, env, input.sounding.slot);
        end

        for m = 1:M
            timer = tic;
            [states{m}, record] = methods(m).step(states{m}, ctx, input, empty_record(U));
            record.runtimeMs = 1000 * toc(timer);
            result.metrics{m} = store_record(result.metrics{m}, record, t);
        end

    end

    for m = 1:M
        ctrl = states{m}.ctrl;
        result.metrics{m}.control = struct('messages', ctrl.messages, 'bits', ctrl.bits, ...
            'dropped', ctrl.dropped, 'late', ctrl.late, 'evaluations', states{m}.evaluations);
    end

end

function method = method_for(name)

    switch name
        case {'canonical', 'canonical-ue'}
            method = method_canonical();
        case {'nr-oracle', 'nr-csi', 'nr-pilot', 'hold'}
            method = method_codebook();
        case 'cov-pilot'
            method = method_covariance();
        case {'ue-pilot', 'ue-pilot-free'}
            method = method_reciprocity();
        otherwise
            error('unknown method %s', name);
    end

end

function ctx = run_context(cfg, env, spec, realization)
    spec.theta = spec.theta(:);
    ctx = struct('cfg', cfg, 'env', env, 'objective', spec, 'policy', cfg.policy, ...
        'modeB', cfg.modeB, 'ue', cfg.ue);
    ctx.selection = struct('theta', spec.theta, 'cb', cfg.cb, 'Lxu', env.Lxu, ...
        'pcap', env.pcap, 'etotal', inf, 'toneWeights', ones(1, env.numGroups) / env.numGroups);

    switch spec.type
        case 'maxresmac'
            ctx.selection.etotal = get_opt(spec, 'sumFraction', 1) * sum(env.pcap);
        case 'maxrmac'
        otherwise
            error('unknown objective %s', spec.type);
    end

    ctx.selectOpts = struct('powerGridDb', 0:-1:cfg.power.floorDb, ...
        'tpcStepsDb', env.dci.tpcStepsDb, 'maxSweeps', cfg.policy.maxSweeps);
    [ctx.slots, ctx.csirsSlots, ctx.srsSlots] = tdd_timeline(cfg, env);
    ctx.calibration = calibration_errors(cfg, env, realization);
end

function [slots, csirsSlots, srsSlots] = tdd_timeline(cfg, env)
    %TDD_TIMELINE Uplink slots of the horizon with the CSI-RS and SRS slots tied to each
    %
    % The CSI-RS slot is the first downlink or special slot at least lagSlots
    % after its uplink slot; the SRS slot is the latest sounding slot since
    % the previous uplink slot, NaN when there is none
    pattern = env.tdd.pattern;
    period = numel(pattern);
    numbers = 0:cfg.horizonSlots + 2 * period + cfg.csirs.lagSlots;
    types = pattern(mod(numbers, period) + 1);
    slots = numbers(types == 'U' & numbers < cfg.horizonSlots);
    downlink = numbers(types ~= 'U');
    sounding = numbers(mod(numbers - cfg.srs.offset, cfg.srs.periodSlots) == 0);
    previous = [-inf, slots(1:end - 1)];
    csirsSlots = zeros(size(slots));
    srsSlots = nan(size(slots));

    for t = 1:numel(slots)
        csirsSlots(t) = downlink(find(downlink >= slots(t) + cfg.csirs.lagSlots, 1));
        recent = sounding(sounding > previous(t) & sounding <= slots(t));

        if ~isempty(recent)
            srsSlots(t) = recent(end);
        end

    end

end

function calibration = calibration_errors(cfg, env, realization)
    %CALIBRATION_ERRORS Residual per-antenna downlink-over-uplink chain gains of a realization
    stream = RandStream('mt19937ar', 'Seed', cfg.seed + realization + 100003);
    spread = cfg.csirs.calibration;
    draw = @(n) 10 .^ (spread.amplitudeDb * randn(stream, n, 1) / 20) .* ...
        exp(1j * deg2rad(spread.phaseDeg) * randn(stream, n, 1));
    calibration = struct('gnb', draw(cfg.numRx), 'ue', {cell(1, env.U)});

    for u = 1:env.U
        calibration.ue{u} = draw(env.Lxu(u));
    end

end

function truth = true_gram(H, env, cfg)
    channel = pagemldivide(env.noiseFactor, H(:, :, :, cfg.evalSymbol));
    truth = struct('Href', channel, 'Keval', pagemtimes(channel, 'ctranspose', channel, 'none'));
    truth.Ksel = truth.Keval(:, :, env.selectSubcarriers);
end

function record = empty_record(U)
    record = struct('rates', zeros(U, 1), 'energy', zeros(U, 1), 'objective', 0, ...
        'csirsRE', 0, 'deployedRow', zeros(U, 1), 'deployedPower', zeros(U, 1), ...
        'throughput', zeros(U, 1), 'planned', nan(U, 1), 'designBits', 0, ...
        'dciBits', 0, 'messages', 0, 'lost', 0, 'nack', 0, 'dataFraction', 1, ...
        'runtimeMs', 0);
end

function metrics = empty_metrics(U, T)
    template = empty_record(U);
    names = fieldnames(template);
    metrics = struct();

    for i = 1:numel(names)
        metrics.(names{i}) = repmat(template.(names{i}), 1, T);
    end

    metrics.throughput(:) = nan;
end

function metrics = store_record(metrics, record, t)
    names = fieldnames(record);

    for i = 1:numel(names)
        metrics.(names{i})(:, t) = record.(names{i});
    end

end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
