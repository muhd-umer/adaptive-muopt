function env = nr_setup(cfg)
    %NR_SETUP Resource grid, reference-signal mappings, and action catalogs of an NR experiment
    %
    % SRS, DMRS, and CSI-RS symbols and resource-element positions come from
    % the 5G Toolbox, so every learner observes exactly the REs a gNB or UE
    % would. SRS ports share a comb through cyclic shifts as long as the comb
    % has room, and DMRS ports fill the CDM groups of the smallest DMRS type
    % that holds every layer; the UEs on one comb or in one CDM group are
    % observed jointly
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %
    % Returns:
    %   env: Carrier, TDD pattern, grid sizes, PRG groups, reference-signal
    %     mappings, codebook catalogs, DCI fields, and overhead

    if exist('nrPUSCHCodebook', 'file') ~= 2 || exist('nrCDLChannel', 'file') ~= 2
        error('nr_setup:toolbox', 'the NR experiments need the 5G Toolbox');
    end

    U = numel(cfg.Lxu);

    fixedGains = isempty(cfg.userGainDb) || numel(cfg.userGainDb) == U;

    if numel(cfg.theta) ~= U || ~fixedGains || ~all(ismember(cfg.Lxu, [1 2 4]))
        error('nr_setup:ues', ['theta and a nonempty userGainDb need one entry per UE, ' ...
            'with 1, 2, or 4 ports']);
    end

    if cfg.numRx ~= prod(cfg.gnbArray)
        error('nr_setup:array', 'numRx must equal prod(gnbArray)');
    end

    env = struct();
    env.U = U;
    env.numRx = cfg.numRx;
    env.Lxu = cfg.Lxu;
    env.Ltot = sum(cfg.Lxu);
    ends = cumsum(cfg.Lxu);
    env.ranges = arrayfun(@(u) ends(u) - cfg.Lxu(u) + 1:ends(u), 1:U, 'UniformOutput', false);
    env.carrier = nrCarrierConfig('NSizeGrid', cfg.carrier.NRB, ...
        'SubcarrierSpacing', cfg.carrier.scs);
    env.ofdm = nrOFDMInfo(env.carrier);
    env.numSubcarriers = 12 * cfg.carrier.NRB;
    env.numSymbols = env.carrier.SymbolsPerSlot;
    env.slotDuration = 1e-3 / (cfg.carrier.scs / 15);
    env.tdd = tdd_pattern(cfg, env);
    env.pcap = 10 ^ (cfg.snrDb / 10) * ones(U, 1);

    groupSize = 12 * cfg.prgRB;
    env.numGroups = env.numSubcarriers / groupSize;

    if env.numGroups ~= fix(env.numGroups)
        error('nr_setup:prg', 'the carrier must hold a whole number of PRGs');
    end

    env.groups = arrayfun(@(g) (g - 1) * groupSize + (1:groupSize), 1:env.numGroups, ...
        'UniformOutput', false);
    env.selectSubcarriers = 1:cfg.selectStride:env.numSubcarriers;
    env.srs = srs_mapping(cfg, env);
    env.dmrs = dmrs_mapping(cfg, env);
    env.csirs = csirs_mapping(cfg, env);
    env.catalogs = cell(1, U);

    for u = 1:U
        opts = cfg.codebook;

        if isempty(opts.maxRank)
            opts.maxRank = cfg.Lxu(u);
        end

        env.catalogs{u} = nr_catalog(cfg.Lxu(u), opts);
        env.defaultRow(u) = find([env.catalogs{u}.rows.rank] == cfg.Lxu(u) & ...
            [env.catalogs{u}.rows.tpmi] == 0, 1);
    end

    env.dci = nr_dci(env.catalogs, cfg.Lxu, cfg.carrier.NRB, env.dmrs.type);
    env.dmrsSymbols = numel(env.dmrs.symbols);
    env.srsSymbols = numel(env.srs.symbols);

    if ~isempty(cfg.noise.covariance)
        env.noiseFactor = chol(cfg.noise.covariance, 'lower');
    else
        env.noiseFactor = eye(env.numRx);
    end

end

function tdd = tdd_pattern(cfg, env)
    %TDD_PATTERN Slot types of the TDD pattern and the downlink REs it offers per second
    pattern = upper(cfg.tdd.pattern);

    if ~all(ismember(pattern, 'DSU')) || ~any(pattern == 'U') || all(pattern == 'U')
        error('nr_setup:tdd', 'the TDD pattern needs uplink and downlink slots');
    end

    symbols = nnz(pattern == 'D') * env.numSymbols + nnz(pattern == 'S') * cfg.tdd.special(1);
    tdd = struct('pattern', pattern, 'downlinkRE', env.numSubcarriers * symbols / ...
        (numel(pattern) * env.slotDuration));

    for slot = cfg.srs.offset:cfg.srs.periodSlots:cfg.srs.offset + lcm(numel(pattern), ...
            cfg.srs.periodSlots)
        type = pattern(mod(slot, numel(pattern)) + 1);

        if type == 'D' || (type == 'S' && cfg.srs.symbolStart < env.numSymbols - cfg.tdd.special(3))
            error('nr_setup:tdd', 'SRS slot %d has no uplink symbols for the SRS', slot);
        end

    end

end

function srs = srs_mapping(cfg, env)
    %SRS_MAPPING Comb, comb offset, and cyclic shift of every UE's SRS
    %
    % Comb 2 offers 8 cyclic shifts and comb 4 offers 12; a UE with L ports
    % takes shifts spaced by Nmax / L. The smallest comb that fits every
    % port on one comb offset keeps the sounding joint; otherwise the UEs fill
    % the comb-4 offsets in turn
    U = env.U;
    combs = [2 8; 4 12];
    choice = find(sum(cfg.Lxu) <= combs(:, 2), 1);

    if isempty(choice)
        choice = 2;
    end

    ktc = combs(choice, 1);
    shiftsMax = combs(choice, 2);
    used = cell(1, ktc);
    placement = zeros(U, 2);

    for u = 1:U
        step = shiftsMax / cfg.Lxu(u);
        placed = false;

        for offset = 1:ktc

            for shift = 0:step - 1
                taken = mod(shift + step * (0:cfg.Lxu(u) - 1), shiftsMax);

                if ~any(ismember(taken, used{offset}))
                    used{offset} = [used{offset}, taken];
                    placement(u, :) = [offset - 1, shift];
                    placed = true;
                    break;
                end

            end

            if placed
                break;
            end

        end

        if ~placed
            error('nr_setup:srs', 'the SRS ports of all UEs do not fit on comb %d', ktc);
        end

    end

    srs = struct('ktc', ktc, 'values', {cell(1, U)}, 'portEnergy', env.pcap(:) * ktc ./ cfg.Lxu(:));
    offsets = unique(placement(:, 1))';
    srs.groups = struct('ues', {}, 'positions', {}, 'groupOf', {});

    for offset = offsets
        members = find(placement(:, 1) == offset)';
        positions = [];

        for u = members
            config = nrSRSConfig('NumSRSPorts', cfg.Lxu(u), 'NumSRSSymbols', cfg.srs.symbols, ...
                'SymbolStart', cfg.srs.symbolStart, 'KTC', ktc, 'KBarTC', offset, ...
                'CyclicShift', placement(u, 2), 'CSRS', cfg.srs.csrs, 'SRSPeriod', [1 0]);
            carrier = env.carrier;
            carrier.NSlot = 0;
            indices = double(nrSRSIndices(carrier, config, 'IndexStyle', 'subscript', ...
                'IndexBase', '1based'));
            srs.values{u} = unit_power(nrSRS(carrier, config));
            mine = indices(indices(:, 3) == 1, 1:2);

            if isempty(positions)
                positions = mine;
            elseif ~isequal(mine, positions)
                error('nr_setup:srs', 'UEs on one comb offset must sound the same REs');
            end

        end

        srs.groups(end + 1) = struct('ues', members, 'positions', positions, ...
            'groupOf', group_of(positions(:, 1), env));
    end

    srs.symbols = unique(srs.groups(1).positions(:, 2))';
end

function dmrs = dmrs_mapping(cfg, env)
    %DMRS_MAPPING DMRS type, CDM group, and ports of every UE, and their sequences
    %
    % Each UE keeps all of its ports in one CDM group, filled largest UE
    % first; a rank-r transmission uses the first r ports. Every CDM group is
    % left without data, which sets the DMRS power boost (TS 38.214 Table
    % 6.2.2-1)
    U = env.U;
    groupPorts = {{[0 1 4 5], [2 3 6 7]}, {[0 1 6 7], [2 3 8 9], [4 5 10 11]}};
    type = find(sum(cfg.Lxu) <= [8 12], 1);

    if isempty(type)
        error('nr_setup:dmrs', 'more than 12 layers exceed the orthogonal DMRS ports');
    end

    ports = groupPorts{type};
    free = 4 * ones(1, numel(ports));
    cdmOf = zeros(1, U);
    [~, sequence] = sort(cfg.Lxu, 'descend');

    for u = sequence
        group = find(free >= cfg.Lxu(u), 1);

        if isempty(group)
            error('nr_setup:dmrs', 'the UE ports do not pack into the DMRS CDM groups');
        end

        cdmOf(u) = group;
        free(group) = free(group) - cfg.Lxu(u);
    end

    dmrs = struct('type', type, 'cdmOf', cdmOf, 'boost', sqrt(numel(ports)), ...
        'ports', {cell(1, U)}, 'values', {cell(U, max(cfg.Lxu))});
    dmrs.cdm = struct('ues', {}, 'positions', {}, 'groupOf', {});

    for c = unique(cdmOf)
        members = find(cdmOf == c);
        start = 0;
        positions = [];

        for u = members
            dmrs.ports{u} = ports{c}(start + (1:cfg.Lxu(u)));
            start = start + cfg.Lxu(u);

            for rank = 1:cfg.Lxu(u)
                pusch = nrPUSCHConfig('NumLayers', rank, 'NumAntennaPorts', cfg.Lxu(u), ...
                    'PRBSet', 0:cfg.carrier.NRB - 1, 'TransmissionScheme', 'nonCodebook');
                pusch.DMRS.DMRSConfigurationType = type;
                pusch.DMRS.DMRSAdditionalPosition = cfg.dmrs.additionalPosition;
                pusch.DMRS.DMRSLength = cfg.dmrs.length;
                pusch.DMRS.NumCDMGroupsWithoutData = numel(ports);
                pusch.DMRS.DMRSPortSet = dmrs.ports{u}(1:rank);
                values = unit_power(nrPUSCHDMRS(env.carrier, pusch));
                indices = double(nrPUSCHDMRSIndices(env.carrier, pusch, ...
                    'IndexStyle', 'subscript', 'IndexBase', '1based'));
                mine = indices(1:size(values, 1), 1:2);

                if isempty(positions)
                    positions = mine;
                elseif ~isequal(mine, positions)
                    error('nr_setup:dmrs', 'the ports of one CDM group must share REs');
                end

                dmrs.values{u, rank} = values;
            end

        end

        dmrs.cdm(end + 1) = struct('ues', members, 'positions', positions, ...
            'groupOf', group_of(positions(:, 1), env));
    end

    dmrs.symbols = unique(dmrs.cdm(1).positions(:, 2))';
end

function csirs = csirs_mapping(cfg, env)
    %CSIRS_MAPPING NZP CSI-RS resources of the reciprocity loop and the CSI-IM
    %
    % One cell-wide resource has a port per gNB antenna; resource pair{u, j}
    % has a port per antenna of UE j and carries UE u's update for adding UE
    % j to its receiver covariance. All use density one and the FD-CDM2 row of
    % their port count (TS 38.211 Table 7.4.1.5.3-1) and are packed with the
    % CSI-IM into the downlink symbols without sharing an RE. The noise gain
    % of the UE's estimator on each resource is measured once on pure noise,
    % which a UE knows of its own receiver
    U = env.U;
    pattern = env.tdd.pattern;
    period = numel(pattern);
    last = env.numSymbols - 1;

    for slot = find(pattern == 'U') - 1
        later = slot + cfg.csirs.lagSlots + (0:2 * period);
        types = pattern(mod(later, period) + 1);

        if types(find(types ~= 'U', 1)) == 'S'
            last = min(last, cfg.tdd.special(1) - 1);
        end

    end

    occupied = false(12, env.numSymbols);
    stream = RandStream('mt19937ar', 'Seed', 4093);
    [csirs.base, occupied] = place(env.numRx, occupied, cfg, env, last, stream);
    csirs.pair = cell(U, U);

    for u = 1:U

        for j = setdiff(1:U, u)
            [csirs.pair{u, j}, occupied] = place(env.Lxu(j), occupied, cfg, env, last, stream);
        end

    end

    [~, occupied] = place(4, occupied, cfg, env, last, []);
    csirs.imRE = 4 * cfg.carrier.NRB;
    csirs.occupied = occupied;
end

function [resource, occupied] = place(ports, occupied, cfg, env, last, stream)
    %PLACE First free position of a CSI-RS resource and its observation layout
    rows = containers.Map({1, 2, 4, 8, 12, 16, 24, 32}, {2, 3, 4, 6, 9, 11, 13, 16});
    counts = containers.Map({1, 2, 4, 8, 12, 16, 24, 32}, {1, 1, 1, 4, 6, 4, 3, 4});

    if ~isKey(rows, ports)
        error('nr_setup:csirs', 'no CSI-RS row has %d ports', ports);
    end

    row = rows(ports);
    numK = counts(ports);
    width = 2 - (row == 2) + 2 * (row == 4);
    pair = ismember(row, [11 13 16]);
    split = ismember(row, [13 16]);
    found = false;

    for l0 = cfg.csirs.symbolStart:last

        symbols = l0 + (0:pair);

        if split
            symbols = [symbols, l0 + 2 + (0:1)];
        end

        if symbols(end) > last
            break;
        end

        for k0 = 0:2:12 - width * numK
            subcarriers = k0 + (0:width * numK - 1);

            if row == 4 && mod(k0, 4) ~= 0
                continue;
            end

            if ~any(occupied(subcarriers + 1, symbols + 1), 'all')
                found = true;
                break;
            end

        end

        if found
            break;
        end

    end

    if ~found
        error('nr_setup:csirs', 'the CSI-RS resources do not fit in the downlink symbols');
    end

    occupied(subcarriers + 1, symbols + 1) = true;
    resource = [];

    if isempty(stream)
        return;
    end

    locations = {l0};

    if split
        locations = {[l0, l0 + 2]};
    end

    ks = k0 + width * (0:numK - 1);

    if numK == 1
        ks = k0;
    end

    config = nrCSIRSConfig('CSIRSType', {'nzp'}, 'CSIRSPeriod', 'on', 'RowNumber', row, ...
        'Density', {'one'}, 'SymbolLocations', locations, 'SubcarrierLocations', {ks}, ...
        'NumRB', cfg.carrier.NRB);
    carrier = env.carrier;
    carrier.NSlot = 0;
    linear = double(nrCSIRSIndices(carrier, config));
    subscripts = double(nrCSIRSIndices(carrier, config, 'IndexStyle', 'subscript', ...
        'IndexBase', '1based'));

    if ~isequal(unique(mod(subscripts(:, 1) - 1, 12))', subcarriers) || ...
            ~isequal(unique(subscripts(:, 2) - 1)', symbols)
        error('nr_setup:csirs', 'CSI-RS row %d does not occupy the planned REs', row);
    end

    values = nrCSIRS(carrier, config);
    values = values ./ abs(values);
    cdm = [2 - (row == 2), 1];
    [cells, ~, position] = unique(sub2ind([env.numSubcarriers, env.numSymbols], ...
        subscripts(:, 1), subscripts(:, 2)));
    portSymbols = zeros(numel(cells), ports);
    portSymbols(sub2ind(size(portSymbols), position, subscripts(:, 3))) = values;
    [k, ~] = ind2sub([env.numSubcarriers, env.numSymbols], cells);
    resource = struct('ports', ports, 'cdm', cdm, 'linear', linear, 'values', values, ...
        'cells', cells, 'portSymbols', portSymbols, 'cellGroup', group_of(k, env), ...
        'numRE', numel(cells));
    resource.reads = cell(1, ports);

    for p = 1:ports
        mine = subscripts(:, 3) == p;
        occasion = (ceil(subscripts(mine, 1) / 12) - 1) * env.numSymbols + subscripts(mine, 2);
        [~, ~, label] = unique(occasion);
        spots = sub2ind([env.numSubcarriers, env.numSymbols], subscripts(mine, 1), ...
            subscripts(mine, 2));
        resource.reads{p} = reshape(accumarray(label, spots, [], @(x) {sort(x)}), 1, []);
    end

    first = cellfun(@(x) x(1), resource.reads{1});
    [k, ~] = ind2sub([env.numSubcarriers, env.numSymbols], first);
    resource.readGroup = group_of(k, env)';
    resource.noiseGain = noise_gain(env, cfg, resource, stream);
end

function gain = noise_gain(env, cfg, resource, stream)
    %NOISE_GAIN Output variance of the UE's CSI-RS estimator per unit input noise
    total = 0;
    entries = 0;

    for trial = 1:10
        grid = (randn(stream, env.numSubcarriers, env.numSymbols) + ...
            1j * randn(stream, env.numSubcarriers, env.numSymbols)) / sqrt(2);
        estimate = nrChannelEstimate(env.carrier, grid, resource.linear, resource.values, ...
            'CDMLengths', resource.cdm, 'AveragingWindow', cfg.csirs.averaging, ...
            'PRGBundleSize', cfg.prgRB, 'Interpolation', 'off');

        for p = 1:resource.ports
            plane = estimate(:, :, 1, p);
            reads = cellfun(@(x) mean(plane(x)), resource.reads{p});
            total = total + sum(abs(reads) .^ 2);
            entries = entries + numel(reads);
        end

    end

    gain = total / entries;
end

function values = unit_power(values)
    values = values ./ sqrt(mean(abs(values) .^ 2, 1));
end

function groupOf = group_of(subcarriers, env)
    groupOf = ceil(subcarriers / (env.numSubcarriers / env.numGroups));
end
