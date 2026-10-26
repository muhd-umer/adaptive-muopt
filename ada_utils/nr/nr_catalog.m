function catalog = nr_catalog(numPorts, opts)
    %NR_CATALOG Codebook-based PUSCH transmit actions of one UE configuration
    %
    % Rows follow the "Precoding information and number of layers" field of
    % DCI format 0_1 for CP-OFDM (TS 38.212 Tables 7.3.1.1.2-2 to -5): the
    % non-coherent entries come first, then partial-coherent, then fully
    % coherent, each listed by rank and then TPMI, keeping only what maxRank
    % and the configured codebookSubset allow. The precoders are taken from
    % nrPUSCHCodebook, and the field width is checked against the tables
    %
    % Args:
    %   numPorts: Number of UE antenna ports, one, two, or four
    %   opts: Optional maxRank (default numPorts), codebookSubset
    %     ('fullyAndPartialAndNonCoherent', 'partialAndNonCoherent', or
    %     'nonCoherent'), and powerMode ('fullpower' for Rel-16
    %     ul-FullPowerTransmission, 'rel15' for the Rel-15 scaling by the share
    %     of ports with nonzero power)
    %
    % Returns:
    %   catalog: Rows with rank, TPMI, and coherence class, the unit-energy
    %     precoders W, the transmit factors F with R = p * F * F', the energy
    %     factor eta = trace(F * F'), and the precoding field width in bits

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    if ~ismember(numPorts, [1, 2, 4])
        error('nr_catalog supports one, two, or four antenna ports');
    end

    maxRank = get_opt(opts, 'maxRank', numPorts);
    subset = get_opt(opts, 'codebookSubset', 'fullyAndPartialAndNonCoherent');
    powerMode = get_opt(opts, 'powerMode', 'fullpower');

    if maxRank < 1 || maxRank > numPorts || maxRank ~= fix(maxRank)
        error('maxRank must lie between one and numPorts');
    end

    classes = {'nonCoherent', 'partialCoherent', 'fullyCoherent'};

    switch subset
        case 'fullyAndPartialAndNonCoherent'
            allowed = classes;
        case 'partialAndNonCoherent'
            allowed = classes(1:2);
        case 'nonCoherent'
            allowed = classes(1);
        otherwise
            error('unknown codebookSubset %s', subset);
    end

    rows = struct('rank', {}, 'tpmi', {}, 'coherence', {});

    if numPorts == 1
        rows(1) = struct('rank', 1, 'tpmi', 0, 'coherence', 'nonCoherent');
    else

        for c = 1:numel(allowed)

            for rank = 1:maxRank

                for tpmi = tpmi_range(numPorts, rank, allowed{c})
                    rows(end + 1) = struct('rank', rank, 'tpmi', tpmi, ...
                        'coherence', allowed{c}); %#ok<AGROW>
                end

            end

        end

    end

    count = numel(rows);
    catalog = struct();
    catalog.numPorts = numPorts;
    catalog.maxRank = maxRank;
    catalog.codebookSubset = subset;
    catalog.powerMode = powerMode;
    catalog.rows = rows;
    catalog.W = cell(1, count);
    catalog.F = cell(1, count);
    catalog.eta = zeros(1, count);
    catalog.rank = [rows.rank];

    for k = 1:count

        if numPorts == 1
            W = 1;
        else
            W = transpose(nrPUSCHCodebook(rows(k).rank, numPorts, rows(k).tpmi));
        end

        catalog.W{k} = W;
        active = any(abs(W) > 0, 2);

        switch powerMode
            case 'fullpower'
                eta = 1;
            case 'rel15'
                eta = nnz(active) / numPorts;
            otherwise
                error('unknown powerMode %s', powerMode);
        end

        catalog.F{k} = sqrt(eta) * W / norm(W, 'fro');
        catalog.eta(k) = eta;
    end

    catalog.bits = ceil(log2(count));
    expected = table_width(numPorts, maxRank, subset);

    if catalog.bits ~= expected
        error('precoding field of %d bits disagrees with the %d-bit TS 38.212 field', ...
            catalog.bits, expected);
    end

end

function values = tpmi_range(numPorts, rank, coherence)
    %TPMI_RANGE TPMIs of one coherence class (TS 38.211 Tables 6.3.1.5-1 to -7)

    if numPorts == 2
        bounds = struct('r1', [0 1; 0 -1; 2 5], 'r2', [0 0; 0 -1; 1 2]);
    else
        bounds = struct('r1', [0 3; 4 11; 12 27], 'r2', [0 5; 6 13; 14 21], ...
            'r3', [0 0; 1 2; 3 6], 'r4', [0 0; 1 2; 3 4]);
    end

    table = bounds.(sprintf('r%d', rank));
    row = find(strcmp(coherence, {'nonCoherent', 'partialCoherent', 'fullyCoherent'}));
    values = table(row, 1):table(row, 2);
end

function bits = table_width(numPorts, maxRank, subset)

    if numPorts == 1
        bits = 0;
        return;
    end

    subsets = {'fullyAndPartialAndNonCoherent', 'partialAndNonCoherent', 'nonCoherent'};
    column = find(strcmp(subset, subsets));

    if numPorts == 2
        widths = [3 0 1; 4 0 2];
        bits = widths(min(maxRank, 2), column);

        if column == 2
            error('two-port codebooks have no partial-coherent subset');
        end

    else
        widths = [5 4 2; 6 5 4];
        bits = widths(1 + (maxRank > 1), column);
    end

end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
