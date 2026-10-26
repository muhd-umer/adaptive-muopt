function message = cov_message(L, G, opts)
    %COV_MESSAGE Per-UE payload for a set of per-band covariances, a nonstandard extension
    %
    % A UE's design is sent as one total-energy field relative to its budget
    % followed by G covariances coded by cov_codec relative to that total.
    % The UE rescales its decoded band energies to the decoded total, so the
    % coarse per-band energy steps redistribute energy instead of losing it.
    % Eigenvector angles are always coded absolutely, since near-equal
    % eigenvalues make their differential coding ill-posed
    %
    % Args:
    %   L: Transmit dimensions of the UE
    %   G: Number of bands, one covariance each
    %   opts: cov_codec options plus totalBits (6) and totalRangeDb (30)
    %
    % Returns:
    %   message: Bit count, band codec, and handles where
    %     [bits, params] = message.encode(R, budget, reference) and
    %     [R, params] = message.decode(bits, budget, reference), with R a 1-by-G
    %     cell, params a 1-by-(G + 1) cell, and reference empty for absolute
    %     coding

    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    message = struct();
    message.L = L;
    message.G = G;
    message.totalBits = get_opt(opts, 'totalBits', 6);
    message.totalRangeDb = get_opt(opts, 'totalRangeDb', 30);
    message.band = cov_codec(L, opts);
    message.angleMask = [false(1, L), true(1, L * (L - 1))];
    message.bits = message.totalBits + G * message.band.bits;
    message.encode = @(R, budget, reference) encode(message, R, budget, reference);
    message.decode = @(bits, budget, reference) decode(message, bits, budget, reference);
end

function [bits, params] = encode(message, R, budget, reference)
    G = message.G;
    total = sum(cellfun(@(B) real(trace(B)), R));
    params = cell(1, G + 1);
    pieces = cell(1, G + 1);

    if total <= 0
        totalDb = -message.totalRangeDb;
    else
        totalDb = min(max(10 * log10(total / budget), -message.totalRangeDb), 0);
    end

    [pieces{1}, params{1}] = scalar_code(totalDb, -message.totalRangeDb, 0, ...
        message.totalBits, reference_part(reference, 1), message.band.deltaFraction);

    for g = 1:G
        [pieces{g + 1}, params{g + 1}] = message.band.encode(R{g}, max(total, realmin), ...
            band_reference(message, reference, g + 1));
    end

    bits = [pieces{:}];
end

function [R, params] = decode(message, bits, budget, reference)
    G = message.G;

    if numel(bits) ~= message.bits
        error('covariance message has %d bits, expected %d', numel(bits), message.bits);
    end

    params = cell(1, G + 1);
    [totalDb, params{1}] = scalar_decode(bits(1:message.totalBits), ...
        -message.totalRangeDb, 0, message.totalBits, reference_part(reference, 1), ...
        message.band.deltaFraction);
    total = budget * 10 ^ (totalDb / 10);
    offset = message.totalBits;
    R = cell(1, G);

    for g = 1:G
        chunk = bits(offset + (1:message.band.bits));
        [R{g}, params{g + 1}] = message.band.decode(chunk, 1, ...
            band_reference(message, reference, g + 1));
        offset = offset + message.band.bits;
    end

    used = sum(cellfun(@(B) real(trace(B)), R));

    if used > 0 && totalDb > -message.totalRangeDb + 1e-12

        for g = 1:G
            R{g} = R{g} * (total / used);
        end

    else

        for g = 1:G
            R{g} = zeros(message.L);
        end

    end

end

function part = band_reference(message, reference, i)
    part = reference_part(reference, i);

    if ~isempty(part)
        part(message.angleMask) = nan;
    end

end

function part = reference_part(reference, i)

    if isempty(reference)
        part = [];
    else
        part = reference{i};
    end

end

function [bits, value] = scalar_code(value, low, high, width, reference, fraction)
    [low, high] = scalar_window(low, high, reference, fraction);
    levels = 2 ^ width;
    index = round(min(max((value - low) / (high - low), 0), 1) * (levels - 1));
    bits = false(1, width);

    for i = 1:width
        bits(i) = bitget(index, width - i + 1);
    end

    value = low + (high - low) * index / (levels - 1);
end

function [value, stored] = scalar_decode(bits, low, high, width, reference, fraction)
    [low, high] = scalar_window(low, high, reference, fraction);
    index = 0;

    for i = 1:width
        index = 2 * index + double(bits(i));
    end

    value = low + (high - low) * index / (2 ^ width - 1);
    stored = value;
end

function [low, high] = scalar_window(low, high, reference, fraction)

    if isempty(reference)
        return;
    end

    half = 0.5 * fraction * (high - low);
    centre = min(max(reference, low + half), high - half);
    low = centre - half;
    high = centre + half;
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
