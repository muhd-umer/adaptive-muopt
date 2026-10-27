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
    state.splitStep = [];
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
                    [b, E, state.split, state.splitStep] = capped_canonical(Href, Lxu, ...
                        etotal, env.pcap(active) * T, spec.theta(active), state.split, ...
                        state.splitStep);
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

function [b, E, split, step] = capped_canonical(H, Lxu, Etotal, caps, theta, split, step)
    %CAPPED_CANONICAL Shared budget with per-UE caps through a maxRMAC budget split
    %
    % The capped problem equals maxRMAC at the optimal split of the shared
    % budget, where the energy prices w of the uncapped UEs agree. Two UEs find
    % the root of w1 - w2; more UEs use spectral projected gradient ascent on
    % the weighted rate, whose gradient in the budgets is w, with
    % Barzilai-Borwein steps, until a step moves no budget by more than 0.3 *
    % Etotal / 1024. Both start from the last slot's split and step
    caps = caps(:);

    if numel(split) ~= numel(caps)
        [split, step] = deal([]);
    end

    if sum(caps) <= Etotal
        [~, E, ~, b] = maxRMACMIMO(H, Lxu, caps, theta, 1, true);
        split = caps / Etotal;
        return;
    end

    if numel(theta) == 2
        [b, E, split] = match_prices(H, Lxu, Etotal, caps, theta, split);
        return;
    end

    if isempty(split)
        split = caps / sum(caps);
    end

    budgets = fit_split(split * Etotal, caps, Etotal);
    [~, E, w, b] = maxRMACMIMO(H, Lxu, budgets, theta, 1, true);
    w = w(:);
    value = theta(:)' * sum(b, 2);

    if isempty(step)
        step = 0.1 * Etotal / max(abs(w));
    end

    for iteration = 1:30
        direction = fit_split(budgets + step * w, caps, Etotal) - budgets;

        if max(abs(direction)) <= 0.3 * Etotal / 1024
            break;
        end

        gain = w' * direction;

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

        move = trial - budgets;
        curvature = -move' * (trialW(:) - w);

        if curvature > 0
            step = min(max((move' * move) / curvature, 1e-6 * step), 1e6 * step);
        else
            step = 2 * step;
        end

        [budgets, E, w, b, value] = deal(trial, trialE, trialW(:), trialB, trialValue);
    end

    split = budgets / Etotal;
end

function [b, E, split] = match_prices(H, Lxu, Etotal, caps, theta, split)
    %MATCH_PRICES Two-UE budget split where the energy prices agree, by Illinois regula falsi
    %
    % The price difference w1 - w2 falls as UE 1's budget grows. A bracket
    % of width 0.3 * Etotal around the last split is widened to the feasible
    % range when it misses the root, and the root is resolved to 0.3 *
    % Etotal / 1024 or until the prices agree
    low = max(0, Etotal - caps(2));
    high = min(caps(1), Etotal);
    solve = @(x) split_point(H, Lxu, Etotal, theta, x);

    if isempty(split)
        guess = 0.5 * (low + high);
    else
        guess = min(max(split(1) * Etotal, low), high);
    end

    left = solve(max(low, guess - 0.15 * Etotal));
    right = solve(min(high, guess + 0.15 * Etotal));

    if left.f < 0 && left.x > low
        right = left;
        left = solve(low);
    elseif right.f > 0 && right.x < high
        left = right;
        right = solve(high);
    end

    if left.f <= 0
        point = left;
    elseif right.f >= 0
        point = right;
    else
        [point, side] = deal(left, 0);
        [fLeft, fRight] = deal(left.f, right.f);

        while right.x - left.x > 0.3 * Etotal / 1024
            x = right.x - fRight * (right.x - left.x) / (fRight - fLeft);
            gap = 0.01 * (right.x - left.x);
            point = solve(min(max(x, left.x + gap), right.x - gap));

            if point.f > 0
                [left, fLeft] = deal(point, point.f);

                if side == 1
                    fRight = fRight / 2;
                end

                side = 1;
            else
                [right, fRight] = deal(point, point.f);

                if side == -1
                    fLeft = fLeft / 2;
                end

                side = -1;
            end

            if abs(point.f) <= 1e-6 * max(abs(point.w))
                break;
            end

        end

    end

    [b, E] = deal(point.b, point.E);
    split = [point.x; Etotal - point.x] / Etotal;
end

function point = split_point(H, Lxu, Etotal, theta, x)
    [~, E, w, b] = maxRMACMIMO(H, Lxu, [x; Etotal - x], theta, 1, true);
    point = struct('x', x, 'f', w(1) - w(2), 'w', w, 'E', E, 'b', b);
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
