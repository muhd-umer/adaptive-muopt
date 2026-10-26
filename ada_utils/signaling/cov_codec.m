function codec = cov_codec(L, opts)
    %COV_CODEC Quantizer for continuous transmit covariances, a nonstandard extension
    %
    % A covariance R = E * V * diag(s) * V' is sent as its energy E in dB
    % below the UE cap, its eigenvalue shares s by stick breaking, and its
    % eigenvectors V by the L*(L-1)/2 Givens rotations (angle, phase) that
    % reduce V to a diagonal phase matrix, which does not affect R. Absolute
    % coding spans the full parameter ranges; differential coding spends the
    % same bits on a narrower window centred on a reference parameter vector,
    % such as the UE's currently deployed design. This is a research
    % extension with a custom payload, not an NR capability
    %
    % Args:
    %   L: Transmit dimensions of the UE
    %   opts: Optional energyBits (4), energyRangeDb (30), shareBits (3),
    %     angleBits (3), phaseBits (4), and differential window fractions
    %     deltaFraction (0.25) of every range; a NaN reference entry codes that
    %     parameter absolutely
    %
    % Returns:
    %   codec: Bit widths, ranges, and encode/decode handles where
    %     [bits, params] = codec.encode(R, cap, reference) and
    %     [R, params] = codec.decode(bits, cap, reference), with reference
    %     empty for absolute coding

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    codec = struct();
    codec.L = L;
    codec.energyBits = get_opt(opts, 'energyBits', 4);
    codec.energyRangeDb = get_opt(opts, 'energyRangeDb', 30);
    codec.shareBits = get_opt(opts, 'shareBits', 3);
    codec.angleBits = get_opt(opts, 'angleBits', 3);
    codec.phaseBits = get_opt(opts, 'phaseBits', 4);
    codec.deltaFraction = get_opt(opts, 'deltaFraction', 0.25);
    numRotations = L * (L - 1) / 2;
    codec.widths = [codec.energyBits, repmat(codec.shareBits, 1, L - 1), ...
        repmat([codec.angleBits, codec.phaseBits], 1, numRotations)];
    codec.low = [-codec.energyRangeDb, zeros(1, L - 1), ...
        repmat([0, -pi], 1, numRotations)];
    % sorted eigenvalue shares cannot break the stick past (L - i) / (L - i + 1)
    codec.high = [0, (L - 1:-1:1) ./ (L:-1:2), repmat([pi / 2, pi], 1, numRotations)];
    codec.periodic = [false, false(1, L - 1), repmat([false, true], 1, numRotations)];
    codec.bits = sum(codec.widths);
    codec.encode = @(R, cap, reference) encode(codec, R, cap, reference);
    codec.decode = @(bits, cap, reference) decode(codec, bits, cap, reference);
    codec.parameters = @(R, cap) parameters(codec, R, cap);
end

function [bits, params] = encode(codec, R, cap, reference)
    params = parameters(codec, R, cap);
    [low, high] = window(codec, reference);
    levels = 2 .^ codec.widths;
    bits = false(1, codec.bits);
    offset = 0;
    quantized = params;

    for i = 1:numel(params)
        value = params(i);

        if codec.periodic(i) && ~isempty(reference) && ~isnan(reference(i))
            value = reference(i) + wrap(value - reference(i));
        end

        position = (value - low(i)) / (high(i) - low(i));

        if codec.periodic(i)
            index = mod(round(position * levels(i)), levels(i));
        else
            index = round(min(max(position, 0), 1) * (levels(i) - 1));
        end

        bits(offset + (1:codec.widths(i))) = to_bits(index, codec.widths(i));
        quantized(i) = level_value(codec, i, index, low, high);
        offset = offset + codec.widths(i);
    end

    params = quantized;
end

function [R, params] = decode(codec, bits, cap, reference)

    if numel(bits) ~= codec.bits
        error('covariance message has %d bits, codec expects %d', numel(bits), codec.bits);
    end

    [low, high] = window(codec, reference);
    params = zeros(1, numel(codec.widths));
    offset = 0;

    for i = 1:numel(codec.widths)
        index = from_bits(bits(offset + (1:codec.widths(i))));
        params(i) = level_value(codec, i, index, low, high);
        offset = offset + codec.widths(i);
    end

    R = rebuild(codec, params, cap);
end

function value = level_value(codec, i, index, low, high)
    levels = 2 ^ codec.widths(i);

    if codec.periodic(i)
        value = low(i) + (high(i) - low(i)) * index / levels;
    else
        value = low(i) + (high(i) - low(i)) * index / max(levels - 1, 1);
    end

end

function [low, high] = window(codec, reference)
    low = codec.low;
    high = codec.high;

    if isempty(reference)
        return;
    end

    half = 0.5 * codec.deltaFraction * (codec.high - codec.low);
    centre = min(max(reference, codec.low + half), codec.high - half);
    centre(codec.periodic) = reference(codec.periodic);
    absolute = isnan(reference);
    low = centre - half;
    high = centre + half;
    low(absolute) = codec.low(absolute);
    high(absolute) = codec.high(absolute);
end

function params = parameters(codec, R, cap)
    L = codec.L;
    R = 0.5 * (R + R');
    [V, D] = eig(R);
    [values, order] = sort(max(real(diag(D)), 0), 'descend');
    V = V(:, order);
    energy = sum(values);

    if energy <= 0
        params = [codec.low(1), zeros(1, L - 1), zeros(1, L * (L - 1))];
        return;
    end

    energyDb = 10 * log10(energy / cap);
    remaining = 1;
    shares = zeros(1, L - 1);
    fractions = values / energy;

    for i = 1:L - 1
        shares(i) = min(max(1 - fractions(i) / max(remaining, realmin), 0), 1);
        remaining = remaining - fractions(i);
    end

    params = [min(max(energyDb, codec.low(1)), 0), shares, givens_angles(V)];
end

function R = rebuild(codec, params, cap)
    L = codec.L;
    energy = cap * 10 ^ (params(1) / 10);

    if params(1) <= codec.low(1) + 1e-12
        energy = 0;
    end

    fractions = zeros(L, 1);
    remaining = 1;

    for i = 1:L - 1
        fractions(i) = remaining * (1 - params(1 + i));
        remaining = remaining - fractions(i);
    end

    fractions(L) = max(remaining, 0);
    V = givens_unitary(params(L + 1:end), L);
    R = energy * V * diag(fractions) * V';
    R = 0.5 * (R + R');
end

function angles = givens_angles(V)
    %GIVENS_ANGLES Rotations that reduce a unitary matrix to diagonal phases
    L = size(V, 1);
    angles = zeros(1, L * (L - 1));
    k = 0;

    for column = 1:L - 1

        for row = L:-1:column + 1
            a = V(row - 1, column);
            b = V(row, column);
            angle = atan2(abs(b), abs(a));
            phase = angle_of(b) - angle_of(a);
            G = rotation(angle, phase);
            V([row - 1, row], :) = G' * V([row - 1, row], :);
            k = k + 1;
            angles(2 * k - 1:2 * k) = [angle, wrap(phase)];
        end

    end

end

function V = givens_unitary(angles, L)
    V = eye(L);
    k = 0;

    for column = 1:L - 1

        for row = L:-1:column + 1
            k = k + 1;
            G = eye(L);
            G([row - 1, row], [row - 1, row]) = rotation(angles(2 * k - 1), angles(2 * k));
            V = V * G;
        end

    end

end

function G = rotation(angle, phase)
    c = cos(angle);
    s = sin(angle);
    G = [c, -s * exp(-1j * phase); s * exp(1j * phase), c];
end

function value = angle_of(z)

    if abs(z) == 0
        value = 0;
    else
        value = angle(z);
    end

end

function value = wrap(value)
    value = mod(value + pi, 2 * pi) - pi;
end

function bits = to_bits(value, width)
    bits = false(1, width);

    for i = 1:width
        bits(i) = bitget(value, width - i + 1);
    end

end

function value = from_bits(bits)
    value = 0;

    for i = 1:numel(bits)
        value = 2 * value + double(bits(i));
    end

end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
