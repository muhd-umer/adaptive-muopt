function [state, plan] = nr_recip_policy(state, ctx, dmrs, slot)
    %NR_RECIP_POLICY gNB side of reciprocity-based canonical adaptation (TDD, non-codebook PUSCH)
    %
    % The gNB regresses the DMRS of each CDM group on its known symbols, which
    % yields every UE's effective channel B_j through its own UE-chosen
    % precoder, and pools the residuals of all groups and PRGs into one
    % wideband noise-plus-interference covariance Rn; the channel matrix
    % itself is never formed. UE u needs Q_k = H_u' * inv(S_k without u) * H_u
    % for every nested set k it decodes within. The gNB sends one cell-wide
    % CSI-RS precoded with inv(Rn)^(1/2) and, for each other UE j in decoding
    % order, a CSI-RS with UE j's port count precoded with
    % Y * B_j * inv(I + B_j' * Y * B_j)^(1/2), where Y is the inverse
    % covariance before UE j is added. By the matrix inversion lemma every
    % Q_k is the cell-wide Gram less the Grams of the updates before it, so
    % the CSI-RS load grows with the UE ports rather than with the gNB
    % antennas per term. For maxRESMAC the gNB also steers each UE's PUSCH
    % power with accumulated TPC commands
    %
    % Args:
    %   state: Policy state with the per-UE budgets and the ready flag
    %   ctx: Shared run context with env, objective, and settings
    %   dmrs: DMRS observation of this slot through the UE-chosen precoders
    %   slot: Current slot
    %
    % Returns:
    %   state: Updated budgets
    %   plan: Decoding order, term weights, the cell-wide precoder base, and
    %     per UE the other UEs in the order they are added, the update
    %     precoders, the terms with the number of updates before each, the
    %     learned per-user rates, and the UEs sent a nonzero TPC step

    env = ctx.env;
    spec = ctx.objective;
    U = env.U;
    G = env.numGroups;
    numRx = env.numRx;
    boost = env.dmrs.boost;
    [sorted, order] = sort(spec.theta(:), 'descend');
    delta = sorted - [sorted(2:end); 0];
    plan = struct('order', order(:)', 'delta', delta, 'base', [], ...
        'others', {cell(1, U)}, 'updates', {cell(1, U)}, 'terms', {cell(1, U)}, ...
        'recordAfter', {cell(1, U)}, 'rates', zeros(U, 1), 'tpc', false(U, 1));

    if isempty(dmrs.cdm)
        return;
    end

    fits = cell(numel(dmrs.cdm), G);
    residual = zeros(numRx);
    dof = 0;

    % the effective channel may vary linearly across the PRG, so frequency
    % selectivity is not mistaken for noise
    for c = 1:numel(dmrs.cdm)
        group = dmrs.cdm(c);

        for g = 1:G
            X = group.X{g};
            layers = size(X, 1);
            offsets = reshape(group.offsets{g}, 1, []);
            regressors = [X; X .* offsets];
            normal = regressors * regressors' + 1e-12 * eye(2 * layers);
            coefficients = (group.Y{g} * regressors') / normal;
            misfit = group.Y{g} - coefficients * regressors;
            residual = residual + misfit * misfit';
            dof = dof + size(X, 2) - 2 * layers;
            fits{c, g} = struct('coefficients', coefficients, 'layers', layers, ...
                'sensitivity', real(diag(inv(normal))), 'spread', mean(offsets .^ 2));
        end

    end

    noise = hermitian(residual) / max(dof, 1);
    covariances = repmat({zeros(numRx, numRx, G)}, 1, U);

    % least-squares coefficients carry noise of covariance Rn * inv(Xa * Xa'),
    % which would otherwise inflate every learned received covariance
    for c = 1:numel(dmrs.cdm)
        group = dmrs.cdm(c);

        for g = 1:G
            fit = fits{c, g};

            for u = group.ues
                rows = group.layerRows{u};
                level = fit.coefficients(:, rows) / boost;
                slope = fit.coefficients(:, fit.layers + rows) / boost;
                bias = (sum(fit.sensitivity(rows)) + fit.spread * ...
                    sum(fit.sensitivity(fit.layers + rows))) / boost ^ 2 * noise;
                covariances{u}(:, :, g) = psd(hermitian(level * level' + ...
                    fit.spread * (slope * slope') - bias));
            end

        end

    end

    rates = zeros(U, G);

    for g = 1:G
        S = noise;
        previous = page_logdet(S);

        for k = 1:U
            S = S + covariances{order(k)}(:, :, g);
            current = page_logdet(S);
            rates(order(k), g) = (current - previous) / log(2);
            previous = current;
        end

    end

    plan.rates = mean(rates, 2);
    base = hermitian(inv(noise));
    plan.base = repmat(base, 1, 1, G);
    active = state.active(:)';

    for u = find(active)
        position = find(order == u);
        terms = position:U;
        terms = terms(delta(terms) > 0);
        sequence = order(:)';
        others = sequence(sequence ~= u & active(sequence));
        plan.terms{u} = terms;
        plan.others{u} = others;
        plan.recordAfter{u} = arrayfun(@(k) nnz(ismember(others, sequence(1:k))), terms);
        plan.updates{u} = cell(1, max([plan.recordAfter{u}, 0]));
        inverse = repmat(base, 1, 1, G);

        for m = 1:numel(plan.updates{u})
            j = others(m);
            update = zeros(numRx, env.Lxu(j), G);

            for g = 1:G
                B = dominant_factor(covariances{j}(:, :, g), env.Lxu(j));
                Y = inverse(:, :, g);
                YB = Y * B;
                [V, D] = eig(hermitian(eye(env.Lxu(j)) + B' * YB));
                values = max(real(diag(D)), realmin);
                update(:, :, g) = YB * (V * diag(values .^ -0.5) * V');
                inverse(:, :, g) = hermitian(Y - YB * (V * diag(1 ./ values) * V') * YB');
            end

            plan.updates{u}{m} = update;
        end

    end

    if strcmp(spec.type, 'maxresmac')
        [state, plan] = update_budgets(state, ctx, plan, repmat(noise, 1, 1, G), ...
            covariances, order, delta, slot);
    end

    state.ready = true;
end

function B = dominant_factor(C, count)
    %DOMINANT_FACTOR Factor B with B * B' the best rank-count approximation of C
    [V, D] = eig(hermitian(C));
    [values, sequence] = sort(max(real(diag(D)), 0), 'descend');
    B = V(:, sequence(1:count)) * diag(sqrt(values(1:count)));
end

function marginal = energy_marginals(noise, covariances, order, delta, budget, env)
    %ENERGY_MARGINALS Value of one more unit of energy along each UE's design
    U = env.U;
    G = env.numGroups;
    marginal = zeros(U, 1);

    for g = 1:G
        S = noise(:, :, g);
        inverses = cell(1, U);

        for k = 1:U
            S = S + covariances{order(k)}(:, :, g);
            inverses{k} = inv(0.5 * (S + S'));
        end

        for u = 1:U
            position = find(order == u);
            M = zeros(size(S));

            for k = position:U
                M = M + delta(k) * inverses{k};
            end

            marginal(u) = marginal(u) + real(trace(M * covariances{u}(:, :, g))) / G;
        end

    end

    marginal = marginal ./ max(budget, realmin);
end

function [state, plan] = update_budgets(state, ctx, plan, noise, covariances, order, ...
        delta, slot)
    %UPDATE_BUDGETS Steer the shared energy budget toward equal marginal values with TPC
    %
    % The marginal value of UE u's energy along its current design is
    % trace(M_u * C_u) per unit energy, where M_u is the delta-weighted sum of
    % inverse receiver covariances and C_u the UE's learned received
    % covariance, so no channel matrix is needed. The target split moves
    % multiplicatively toward equal marginals within the caps and the shared
    % budget, and each UE is sent the accumulated TPC step that comes closest
    % to its target without breaking either limit
    env = ctx.env;
    policy = ctx.ue;

    if slot - state.budgetSlot < policy.budgetPeriod
        return;
    end

    state.budgetSlot = slot;
    marginal = energy_marginals(noise, covariances, order, delta, state.budget(:), env);
    active = state.active(:);
    budget = state.budget(:);
    reference = mean(marginal(active));

    if ~any(active) || reference <= 0
        return;
    end

    caps = env.pcap(:);
    target = budget .* exp(policy.budgetStep * min(max(marginal / reference - 1, -1), 1));
    target = fit_budget(target .* active, caps .* active, ctx.selection.etotal);
    steps = env.dci.tpcStepsDb;
    levelDb = 10 * log10(max(budget ./ caps, realmin));
    choice = zeros(env.U, 1);

    for u = find(active(:)')
        reached = min(max(levelDb(u) + steps, ctx.cfg.power.floorDb), 0);
        [~, choice(u)] = min(abs(reached - 10 * log10(max(target(u) / caps(u), realmin))));
    end

    zero = find(steps == 0);
    choice(~active) = zero;
    next = @(c) caps .* 10 .^ (min(max(levelDb + steps(c)', ctx.cfg.power.floorDb), 0) / 10) ...
        .* active;

    while sum(next(choice)) > ctx.selection.etotal * (1 + 1e-9) && any(choice(active) > 1)
        [~, u] = max(choice .* active);
        choice(u) = choice(u) - 1;
    end

    updated = next(choice);
    plan.tpc = active & choice ~= zero & abs(updated - budget) > 1e-12 * caps;
    state.budget = updated;
end

function budget = fit_budget(budget, caps, total)
    %FIT_BUDGET Rescale budgets onto sum = total without exceeding any cap

    for iteration = 1:20
        free = budget < caps * (1 - 1e-12);
        spare = total - sum(budget);

        if abs(spare) <= 1e-12 * total || ~any(free)
            break;
        end

        budget(free) = budget(free) * (1 + spare / max(sum(budget(free)), realmin));
        budget = min(budget, caps);
    end

    budget = min(budget, caps);
end

function matrix = hermitian(matrix)
    matrix = 0.5 * (matrix + matrix');
end

function matrix = psd(matrix)
    [V, D] = eig(matrix);
    matrix = V * diag(max(real(diag(D)), 0)) * V';
    matrix = 0.5 * (matrix + matrix');
end
