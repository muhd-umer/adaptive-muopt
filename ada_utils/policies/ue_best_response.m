function [R, value] = ue_best_response(Q, delta, budget, opts)
    %UE_BEST_RESPONSE A UE's own covariance from the Grams it measured on precoded CSI-RS
    %
    % With the other users held fixed, the canonical objective seen by UE u is
    % sum_k delta_k * logdet(I + R * Q_k) per band, where Q_k is the Gram of
    % its channel against the receiver covariance of the k-th nested user set
    % without itself. That is concave in R, so the UE maximizes it locally by
    % spectral projected gradient over its own power constraint, spending the
    % whole budget, without any covariance feedback
    %
    % Args:
    %   Q: 1-by-K cell of L-by-L-by-G measured Grams, one per nested set
    %   delta: K-by-1 nonnegative term weights
    %   budget: Energy limit per RE averaged over the G bands
    %   opts: Optional R0 (warm start, L-by-L-by-G), flatPsd (true keeps every
    %     band at the same energy, as a flat PUSCH power spectral density
    %     requires), and iterations (40)
    %
    % Returns:
    %   R: L-by-L-by-G covariance
    %   value: Objective in bits per RE

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    L = size(Q{1}, 1);
    G = size(Q{1}, 3);
    flatPsd = get_opt(opts, 'flatPsd', true);
    iterations = get_opt(opts, 'iterations', 40);
    R = get_opt(opts, 'R0', repmat(budget / L * eye(L), 1, 1, G));
    R = project(R, budget, flatPsd);

    if budget <= 0 || ~any(delta > 0)
        R = zeros(L, L, G);
        value = 0;
        return;
    end

    [value, gradient] = objective(Q, delta, R);
    alpha = 1 / max(norm(gradient(:)), eps) * budget;
    previousR = [];
    previousGradient = [];

    for iteration = 1:iterations

        if ~isempty(previousR)
            s = R - previousR;
            y = previousGradient - gradient;
            sy = real(s(:)' * y(:));

            if sy > 0
                alpha = min(max(real(s(:)' * s(:)) / sy, 1e-6 * budget), 1e6 * budget);
            end

        end

        target = project(R + alpha * gradient, budget, flatPsd);
        direction = target - R;
        gain = real(gradient(:)' * direction(:));

        if gain <= 1e-12 * max(1, abs(value))
            break;
        end

        step = 1;

        for backtrack = 1:20
            trial = R + step * direction;
            trialValue = objective(Q, delta, trial);

            if trialValue >= value + 1e-4 * step * gain
                break;
            end

            step = step / 2;
        end

        previousR = R;
        previousGradient = gradient;
        R = trial;
        [value, gradient] = objective(Q, delta, R);
    end

end

function [value, gradient] = objective(Q, delta, R)
    L = size(R, 1);
    G = size(R, 3);
    factor = zeros(L, L, G);

    for g = 1:G
        [V, D] = eig(0.5 * (R(:, :, g) + R(:, :, g)'));
        factor(:, :, g) = V * diag(sqrt(max(real(diag(D)), 0)));
    end

    value = 0;
    gradient = zeros(L, L, G);

    for k = 1:numel(Q)

        if delta(k) == 0
            continue;
        end

        inner = pagemtimes(factor, 'ctranspose', pagemtimes(Q{k}, factor), 'none') + eye(L);
        value = value + delta(k) * sum(page_logdet(inner)) / (G * log(2));

        if nargout > 1
            term = pagemrdivide(Q{k}, eye(L) + pagemtimes(R, Q{k}));
            gradient = gradient + delta(k) * 0.5 * (term + pagectranspose(term)) / (G * log(2));
        end

    end

end

function R = project(R, budget, flatPsd)
    %PROJECT Euclidean projection onto the UE's per-band or pooled energy limit
    L = size(R, 1);
    G = size(R, 3);
    bases = cell(1, G);
    values = zeros(L, G);

    for g = 1:G
        [V, D] = eig(0.5 * (R(:, :, g) + R(:, :, g)'));
        bases{g} = V;
        values(:, g) = real(diag(D));
    end

    if flatPsd

        for g = 1:G
            values(:, g) = waterfill(values(:, g), budget);
        end

    else
        values(:) = waterfill(values(:), budget * G);
    end

    for g = 1:G
        R(:, :, g) = bases{g} * diag(values(:, g)) * bases{g}';
        R(:, :, g) = 0.5 * (R(:, :, g) + R(:, :, g)');
    end

end

function x = waterfill(v, total)
    %WATERFILL Projection onto the simplex {x >= 0, sum(x) = total}
    sorted = sort(v, 'descend');
    cumulative = cumsum(sorted);
    indices = (1:numel(sorted))';
    active = find(sorted - (cumulative - total) ./ indices > 0, 1, 'last');
    threshold = (cumulative(active) - total) / active;
    x = max(v - threshold, 0);
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
