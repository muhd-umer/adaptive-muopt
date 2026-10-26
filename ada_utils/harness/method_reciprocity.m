function method = method_reciprocity()
    %METHOD_RECIPROCITY TDD reciprocity: each UE solves its block of the canonical problem
    %
    % The gNB learns receiver statistics from DMRS and precodes downlink
    % CSI-RS with them (nr_recip_policy); each UE estimates its Grams from that
    % CSI-RS (ue_csirs_gram) and deploys its best response (ue_best_response)
    % as a per-PRG non-codebook precoder. 'ue-pilot' keeps a flat power
    % spectral density, 'ue-pilot-free' may move energy across PRGs. The gNB
    % signals only the SRI when a UE's layer count changes and, for a shared
    % budget, accumulated TPC steps
    %
    % Returns:
    %   method: Handles init(name, ctx, active) and step(state, ctx, input,
    %     record), and the method kind

    method = struct('kind', 'reciprocity', 'init', @init, 'step', @step);
end

function state = init(name, ctx, active)
    state = method_state(name, ctx, active);
    state.flatPsd = ~strcmp(name, 'ue-pilot-free');
    state.R = uniform_covariances(ctx, active);
    state.pending = struct('due', {}, 'ue', {}, 'R', {});
    state.budget = mean(cellfun(@(R) real(trace(R)), state.R), 2);
    state.ranks = ctx.env.Lxu .* active;
    state.budgetSlot = -inf;
    state.ready = false;
end

function [state, record] = step(state, ctx, input, record)
    env = ctx.env;
    G = env.numGroups;
    D = ctx.cfg.control.delay;
    [state, record] = apply_updates(state, ctx, input.active, input.slot, record);

    factors = cov_factors(state.R);
    groupIndex = ceil((1:env.numSubcarriers)' / (env.numSubcarriers / G));
    F = cell(1, env.U);

    for u = 1:env.U
        F{u} = cat(3, factors{u, groupIndex});
    end

    result = mac_eval(input.truth.Keval, F, env.Lxu, ctx.objective.theta, ctx.cfg.cb);
    record.rates = result.userRates;
    record.energy = cellfun(@(R) real(trace(R)), state.R) * ones(G, 1) / G;
    record.objective = result.objective;
    [state, record] = link_outcome(state, ctx, record, input.t);

    layers = layer_factors(state.R, state.active);
    dmrs = nr_dmrs_observe(input.H, env, layers);
    [state, plan] = nr_recip_policy(state, ctx, dmrs, input.slot);
    sri = false(env.U, 1);
    sounded = false;

    for u = find(state.active(:)')

        if isempty(plan.terms{u})
            continue;
        end

        obs = nr_csirs_observe(input.downlink.H, env, ctx.cfg, u, plan.precoders{u}, ...
            ctx.calibration);
        Q = ue_csirs_gram(obs, env, ctx.cfg);
        record.csirsRE = record.csirsRE + numel(Q) * env.csirs.resources{1}.numRE;
        sounded = true;
        best = ue_best_response(Q, plan.delta(plan.terms{u}), state.budget(u), ...
            struct('R0', cat(3, state.R{u, :}), 'flatPsd', state.flatPsd, ...
            'iterations', ctx.ue.iterations));
        state.pending(end + 1) = struct('due', input.downlink.slot + D, 'ue', u, ...
            'R', {squeeze(num2cell(best, [1 2]))'});
        state.evaluations = state.evaluations + 1;
        rank = effective_rank(best);
        sri(u) = rank ~= state.ranks(u);
        state.ranks(u) = rank;
    end

    record.csirsRE = record.csirsRE + sounded * env.csirs.imRE;

    for u = find(sri(:)' | plan.tpc(:)')
        record.designBits = record.designBits + sri(u) * env.dci.sriNonCodebook(u) + ...
            plan.tpc(u) * env.dci.tpc(u);
        record.dciBits = record.dciBits + env.dci.dci01NonCodebook(u);
        record.messages = record.messages + 1;
    end

    record.dataFraction = data_fraction(env, false);
    target = next_occasion(ctx, input.downlink.slot + D);

    if target <= size(state.plan, 2)
        state.plan(:, target) = nr_mcs(max(plan.rates - state.backoff, 0), ...
            cellfun(@(f) size(f, 2), layers)');
    end

end

function [state, record] = apply_updates(state, ctx, active, slot, record)
    %APPLY_UPDATES Deploy the UE updates that have landed and follow scheduling changes
    %
    % An update computed while a UE was scheduled must not revive it after it
    % has been unscheduled, and a scheduling change replaces the updates in
    % flight, which were solved for the old schedule and its budgets
    env = ctx.env;
    due = [state.pending.due] <= slot;

    for i = find(due & active([state.pending.ue]))
        state.R(state.pending(i).ue, :) = state.pending(i).R;
    end

    state.pending = state.pending(~due & active([state.pending.ue]));

    if isequal(active, state.active)
        return;
    end

    fresh = uniform_covariances(ctx, active);
    reset = (active & ~state.active) | ~active;
    state.R(reset, :) = fresh(reset, :);
    state.budget(reset) = mean(cellfun(@(R) real(trace(R)), fresh(reset, :)), 2);
    state.active = active;
    state.pending = state.pending([]);
    [state.R, refit] = fit_shared(state.R, ctx.selection.etotal, active);
    used = mean(cellfun(@(R) real(trace(R)), state.R), 2);
    state.budget(active) = min(used(active), env.pcap(active));

    if sum(state.budget) > ctx.selection.etotal
        state.budget = state.budget * (ctx.selection.etotal / sum(state.budget));
    end

    if refit
        record.messages = record.messages + nnz(active);
    end

end

function R = uniform_covariances(ctx, active)
    %UNIFORM_COVARIANCES Equal-power identity per PRG, within any shared budget
    env = ctx.env;
    energy = min(env.pcap(:) .* active(:), ctx.selection.etotal / max(nnz(active), 1));
    R = cell(env.U, env.numGroups);

    for u = 1:env.U
        R(u, :) = {energy(u) / env.Lxu(u) * eye(env.Lxu(u))};
    end

end

function layers = layer_factors(R, active)
    %LAYER_FACTORS Per-PRG precoders with one wideband layer count, as PUSCH needs
    [U, G] = size(R);
    layers = cell(1, U);

    for u = 1:U
        L = size(R{u, 1}, 1);

        if ~active(u)
            layers{u} = zeros(L, 0);
            continue;
        end

        rank = effective_rank(cat(3, R{u, :}));
        layers{u} = zeros(L, rank, G);

        for g = 1:G
            [V, D] = eig(0.5 * (R{u, g} + R{u, g}'));
            [values, order] = sort(max(real(diag(D)), 0), 'descend');
            layers{u}(:, :, g) = V(:, order(1:rank)) * diag(sqrt(values(1:rank)));
        end

    end

end

function rank = effective_rank(R)
    rank = 1;

    for g = 1:size(R, 3)
        values = sort(max(real(eig(0.5 * (R(:, :, g) + R(:, :, g)'))), 0), 'descend');
        rank = max(rank, nnz(values > 1e-3 * max(sum(values), realmin)));
    end

end
