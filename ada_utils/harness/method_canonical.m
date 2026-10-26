function method = method_canonical()
    %METHOD_CANONICAL Per-slot canonical optimum on the true channel, the genie upper bound
    %
    % 'canonical' solves the unrestricted problem on every subcarrier with no
    % delay. For maxRESMAC, 'canonical-ue' also respects the per-UE caps by
    % splitting the shared budget into maxRMAC budgets
    %
    % Returns:
    %   method: Handles init(name, ctx, active) and step(state, ctx, input,
    %     record), and the method kind

    method = struct('kind', 'canonical', 'init', @init, 'step', @step);
end

function state = init(name, ctx, active)
    state = method_state(name, ctx, active);
    state.split = [];
end

function [state, record] = step(state, ctx, input, record)
    env = ctx.env;
    spec = ctx.objective;
    active = input.active;
    record.dataFraction = data_fraction(env, false);

    if ~any(active)
        return;
    end

    Href = input.truth.Href(:, cell2mat(env.ranges(active)), :);
    T = size(Href, 3);
    Lxu = env.Lxu(active);
    users = find(active);
    rates = zeros(env.U, 1);
    energy = zeros(env.U, 1);

    % the canonical MEX solvers are multi-user, so one scheduled user is
    % solved by closed-form water-filling
    if numel(users) == 1
        [rates(users), energy(users)] = single_user_waterfill(Href, ...
            min(ctx.selection.etotal, env.pcap(users)));
    else

        switch spec.type
            case 'maxresmac'
                etotal = ctx.selection.etotal * T;

                if strcmp(state.name, 'canonical-ue')
                    [b, E, state.split] = capped_canonical(Href, Lxu, etotal, ...
                        env.pcap(active) * T, spec.theta(active), state.split);
                else
                    [~, E, ~, b] = maxRESMACMIMO(Href, Lxu, etotal, spec.theta(active), 1, true);
                end

            case 'maxrmac'
                [~, E, ~, b] = maxRMACMIMO(Href, Lxu, env.pcap(active) * T, ...
                    spec.theta(active), 1, true);
        end

        rates(users) = sum(b, 2) / T;
        energy(users) = sum(E, 2) / T;
    end

    record.rates = rates;
    record.energy = energy;
    record.objective = spec.theta' * rates;
    record.throughput = nr_mcs(rates, env.Lxu(:));
end

function [rate, energy] = single_user_waterfill(H, energy)
    %SINGLE_USER_WATERFILL Per-RE rate of single-user water-filling at an energy budget
    T = size(H, 3);
    gains = [];

    for t = 1:T
        gains = [gains; real(eig(H(:, :, t)' * H(:, :, t)))]; %#ok<AGROW>
    end

    gains = sort(gains(gains > 1e-12), 'descend');

    if isempty(gains)
        rate = 0;
        energy = 0;
        return;
    end

    for count = numel(gains):-1:1
        active = gains(1:count);
        level = (energy * T + sum(1 ./ active)) / count;

        if level > 1 / active(end)
            break;
        end

    end

    rate = sum(log2(level * active)) / T;
    energy = sum(level - 1 ./ active) / T;
end

function [b, E, split] = capped_canonical(H, Lxu, Etotal, caps, theta, split)
    %CAPPED_CANONICAL Shared budget with per-UE caps through a maxRMAC budget split
    %
    % The capped problem equals maxRMAC at the optimal split of the shared
    % budget, where the energy prices w of the uncapped UEs agree. Two UEs
    % bisect on w1 - w2; more UEs use projected gradient ascent on the
    % weighted rate, whose gradient in the budgets is w. Both start from the
    % last slot's split
    caps = caps(:);

    if sum(caps) <= Etotal
        [~, E, ~, b] = maxRMACMIMO(H, Lxu, caps, theta, 1, true);
        split = caps / Etotal;
        return;
    end

    if numel(theta) == 2
        [b, E, split] = bisect_split(H, Lxu, Etotal, caps, theta, split);
        return;
    end

    if isempty(split)
        split = caps / sum(caps);
    end

    budgets = fit_split(split * Etotal, caps, Etotal);
    [~, E, w, b] = maxRMACMIMO(H, Lxu, budgets, theta, 1, true);
    value = theta(:)' * sum(b, 2);
    step = 0.1 * Etotal / max(abs(w(:)));

    for iteration = 1:20
        direction = fit_split(budgets + step * w(:), caps, Etotal) - budgets;
        gain = w(:)' * direction;

        if gain <= 1e-9 * max(abs(value), 1)
            break;
        end

        for backtrack = 1:6
            trial = budgets + direction;
            [~, trialE, trialW, trialB] = maxRMACMIMO(H, Lxu, trial, theta, 1, true);
            trialValue = theta(:)' * sum(trialB, 2);

            if trialValue >= value + 1e-4 * gain
                break;
            end

            direction = direction / 2;
            gain = gain / 2;
        end

        if trialValue < value
            break;
        end

        [budgets, E, w, b, value] = deal(trial, trialE, trialW, trialB, trialValue);
        step = 2 * step;
    end

    split = budgets / Etotal;
end

function [b, E, split] = bisect_split(H, Lxu, Etotal, caps, theta, split)
    low = max(0, Etotal - caps(2));
    high = min(caps(1), Etotal);

    if ~isempty(split)
        width = 0.15 * Etotal;
        lowTrial = max(low, split(1) * Etotal - width);
        highTrial = min(high, split(1) * Etotal + width);
        [~, ~, wLow] = maxRMACMIMO(H, Lxu, [lowTrial; Etotal - lowTrial], theta, 1, true);
        [~, ~, wHigh] = maxRMACMIMO(H, Lxu, [highTrial; Etotal - highTrial], theta, 1, true);

        if wLow(1) >= wLow(2) && wHigh(1) <= wHigh(2)
            low = lowTrial;
            high = highTrial;
        end

    end

    for iteration = 1:10
        middle = 0.5 * (low + high);
        [~, ~, w] = maxRMACMIMO(H, Lxu, [middle; Etotal - middle], theta, 1, true);

        if w(1) > w(2)
            low = middle;
        else
            high = middle;
        end

    end

    middle = 0.5 * (low + high);
    [~, E, ~, b] = maxRMACMIMO(H, Lxu, [middle; Etotal - middle], theta, 1, true);
    split = [middle; Etotal - middle] / Etotal;
end

function x = fit_split(x, caps, total)
    %FIT_SPLIT Projection onto {0 <= x <= caps, sum(x) = total}
    low = min(x - caps) - 1;
    high = max(x);

    for iteration = 1:60
        shift = 0.5 * (low + high);

        if sum(min(max(x - shift, 0), caps)) > total
            low = shift;
        else
            high = shift;
        end

    end

    x = min(max(x - 0.5 * (low + high), 0), caps);
end
