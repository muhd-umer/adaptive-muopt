function out = mac_eval(K, F, Lxu, theta, cb, opts)
    %MAC_EVAL SIC rates, weighted objective, and covariance gradient from a Gram tensor
    %
    % Every quantity is computed from the effective Gram matrix
    % K = H' * inv(Rnn) * H of each tone, so the evaluation is the same whether
    % K comes from a known channel, a channel estimate, or pilot-learned MMSE
    % information. With the decoding order fixed, the weighted rate is
    % sum_k delta_k * logdet(I + F_k' * K_k * F_k) over the nested user sets
    % S_k, and its gradient with respect to R_u is
    % sum_{k >= pos(u)} delta_k * [K_k * inv(I + R_k * K_k)]_uu
    %
    % Args:
    %   K: Ltot-by-Ltot-by-T effective Gram matrices, user streams stacked
    %   F: 1-by-U cell of transmit factors R_u = F_u * F_u', each Lu-by-ru-by-T
    %     or Lu-by-ru when shared by every tone
    %   Lxu: Transmit dimensions per user
    %   theta: Nonnegative user weights or rate prices
    %   cb: Rate scaling, one for complex and two for real baseband
    %   opts: Optional order (users from last to first decoded, default by
    %     descending theta), toneWeights (default uniform average), and
    %     gradient (default false)
    %
    % Returns:
    %   out: Per-user, per-tone rates in bits, tone-weighted user rates,
    %     weighted objective, decoding order, and the per-user gradient cells

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    theta = theta(:);
    U = numel(theta);
    T = size(K, 3);
    Lxu = reshape(Lxu, 1, []);
    order = get_opt(opts, 'order', []);

    if isempty(order)
        [~, order] = sort(theta, 'descend');
    end

    order = order(:)';
    weights = get_opt(opts, 'toneWeights', ones(1, T) / T);
    weights = reshape(weights, 1, []);
    wantGradient = get_opt(opts, 'gradient', false);
    sorted = theta(order);
    delta = sorted - [sorted(2:end); 0];
    ends = cumsum(Lxu);
    starts = [1, ends(1:end - 1) + 1];
    factors = cell(1, U);
    ranks = zeros(1, U);

    for u = 1:U
        factors{u} = F{u};

        if size(factors{u}, 3) == 1 && T > 1
            factors{u} = repmat(factors{u}, 1, 1, T);
        end

        ranks(u) = size(factors{u}, 2);
    end

    out = struct();
    out.order = order;
    out.rates = zeros(U, T);
    logdets = zeros(U, T);
    out.gradient = cell(1, U);

    if wantGradient

        for u = 1:U
            out.gradient{u} = zeros(Lxu(u), Lxu(u), T);
        end

    end

    previous = zeros(1, T);

    for k = 1:U
        members = order(1:k);
        rows = cell2mat(arrayfun(@(u) starts(u):ends(u), members, 'UniformOutput', false));
        Fs = block_factor(factors(members), Lxu(members), ranks(members), T);

        if isempty(Fs) || size(Fs, 2) == 0
            logdets(k, :) = previous;
            out.rates(order(k), :) = 0;
            continue;
        end

        Ks = K(rows, rows, :);
        P = pagemtimes(Ks, Fs);
        M = pagemtimes(Fs, 'ctranspose', P, 'none');
        M = 0.5 * (M + pagectranspose(M)) + eye(size(Fs, 2));
        current = page_logdet(M);
        logdets(k, :) = current;
        out.rates(order(k), :) = (current - previous) / (cb * log(2));
        previous = current;

        if wantGradient && delta(k) ~= 0
            Z = pagemldivide(M, pagectranspose(P));
            G = Ks - pagemtimes(P, Z);
            offset = 0;

            for j = 1:k
                u = members(j);
                span = offset + (1:Lxu(u));
                block = G(span, span, :);
                out.gradient{u} = out.gradient{u} + delta(k) * ...
                    0.5 * (block + pagectranspose(block)) / (cb * log(2));
                offset = offset + Lxu(u);
            end

        end

    end

    out.userRates = out.rates * weights(:);
    out.objective = theta' * out.userRates;
    out.delta = delta;
end

function Fs = block_factor(factors, Lxu, ranks, T)
    Fs = zeros(sum(Lxu), sum(ranks), T);
    rowOffset = 0;
    columnOffset = 0;

    for j = 1:numel(factors)
        Fs(rowOffset + (1:Lxu(j)), columnOffset + (1:ranks(j)), :) = factors{j};
        rowOffset = rowOffset + Lxu(j);
        columnOffset = columnOffset + ranks(j);
    end

end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
