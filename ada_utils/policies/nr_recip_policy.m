function [state, plan] = nr_recip_policy(state, ctx, dmrs, slot)
    %NR_RECIP_POLICY gNB side of reciprocity-based canonical adaptation (TDD, non-codebook PUSCH)
    %
    % The gNB regresses the received DMRS on its known symbols, which yields
    % every UE's effective channel through its own UE-chosen precoder and the
    % residual noise-plus-interference covariance; summing those gives the
    % receiver covariance S_k of every nested user set in the decoding order.
    % These are the receiver-side GDFE statistics; the channel matrix itself is
    % never formed. For UE u and each set k it decodes within, the gNB
    % precodes downlink CSI-RS with inv(S_k without u)^(1/2), so by
    % reciprocity the UE measures H_u' * inv(S_k without u) * H_u and can solve
    % its own block of the canonical problem locally. For maxRESMAC the gNB
    % also splits the shared budget into per-UE PUSCH powers, which it steers
    % with ordinary accumulated TPC commands
    %
    % Args:
    %   state: Policy state with the per-UE budgets and the ready flag
    %   ctx: Shared run context with env, objective, and settings
    %   dmrs: DMRS observation of this slot through the UE-chosen precoders
    %   slot: Current slot
    %
    % Returns:
    %   state: Updated budgets and learned rates
    %   plan: Decoding order, term weights, the per-UE CSI-RS precoding
    %     matrices, learned per-user rates, and the UEs sent a nonzero TPC step

    env = ctx.env;
    spec = ctx.objective;
    U = env.U;
    G = env.numGroups;
    boost = env.dmrsBoost;
    noise = zeros(size(dmrs.Y{1}, 1), size(dmrs.Y{1}, 1), G);
    covariances = cell(1, U);

    for u = 1:U
        covariances{u} = zeros(size(noise));
    end

    for g = 1:G
        Y = dmrs.Y{g};
        X = dmrs.X{g};
        count = size(X, 2);
        layers = size(X, 1);

        % the effective channel is allowed a first-order variation across the
        % PRG, so frequency selectivity is not mistaken for noise
        offsets = reshape(dmrs.offsets{g}, 1, []);
        regressors = [X; X .* offsets];
        coefficients = (Y * regressors') / (regressors * regressors' + ...
            1e-12 * eye(2 * layers));
        residual = Y - coefficients * regressors;
        noise(:, :, g) = hermitian(residual * residual') / max(count - 2 * layers, 1);
        spread = mean(offsets .^ 2);
        % least-squares coefficients carry noise of covariance Rn * inv(Xa * Xa'),
        % which would otherwise inflate every learned received covariance
        sensitivity = real(diag(inv(regressors * regressors' + 1e-12 * eye(2 * layers))));

        for u = 1:U

            if dmrs.ranks(u) > 0
                rows = dmrs.layerRows{u};
                level = coefficients(:, rows) / boost;
                slope = coefficients(:, layers + rows) / boost;
                bias = (sum(sensitivity(rows)) + spread * sum(sensitivity(layers + rows))) / ...
                    boost ^ 2 * noise(:, :, g);
                covariances{u}(:, :, g) = psd(hermitian(level * level' + ...
                    spread * (slope * slope') - bias));
            end

        end

    end

    [sorted, order] = sort(spec.theta(:), 'descend');
    delta = sorted - [sorted(2:end); 0];
    active = state.active(:);
    plan = struct('order', order(:)', 'delta', delta, 'precoders', {cell(1, U)}, ...
        'terms', {cell(1, U)}, 'tpc', false(U, 1));
    rates = zeros(U, G);

    for g = 1:G
        S = noise(:, :, g);
        previous = page_logdet(S);

        for k = 1:U
            S = S + covariances{order(k)}(:, :, g);
            current = page_logdet(S);
            rates(order(k), g) = (current - previous) / log(2);
            previous = current;
        end

    end

    plan.rates = mean(rates, 2);

    for u = find(active(:)')
        position = find(order == u);
        terms = position:U;
        terms = terms(delta(terms) > 0);
        plan.terms{u} = terms;
        plan.precoders{u} = cell(1, numel(terms));

        for i = 1:numel(terms)
            members = order(1:terms(i));
            Xinv = zeros(size(noise));

            for g = 1:G
                S = noise(:, :, g);

                for j = members(members ~= u)
                    S = S + covariances{j}(:, :, g);
                end

                Xinv(:, :, g) = hermitian(inv(S));
            end

            plan.precoders{u}{i} = Xinv;
        end

    end

    if strcmp(spec.type, 'maxresmac')
        [state, plan] = update_budgets(state, ctx, plan, noise, covariances, ...
            order, delta, slot);
    end

    state.learnedRates = plan.rates;
    state.ready = true;
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
