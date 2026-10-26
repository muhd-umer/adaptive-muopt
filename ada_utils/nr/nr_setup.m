function env = nr_setup(cfg)
    %NR_SETUP Resource grid, reference-signal mappings, and action catalogs of an NR experiment
    %
    % SRS, DMRS, and CSI-RS symbols and resource-element positions come from
    % the 5G Toolbox, so every learner observes exactly the REs a gNB or UE
    % would. The SRS of all UEs shares one comb and is separated by cyclic
    % shift, and the DMRS ports of all UEs share one CDM group, so both uplink
    % observations are joint across users
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %
    % Returns:
    %   env: Carrier, TDD pattern, grid sizes, PRG groups, reference-signal
    %     mappings, codebook catalogs, DCI fields, and overhead

    if exist('nrPUSCHCodebook', 'file') ~= 2 || exist('nrSRS', 'file') ~= 2
        error('nr_setup:toolbox', 'the NR experiments need the 5G Toolbox');
    end

    U = numel(cfg.Lxu);
    env = struct();
    env.U = U;
    env.Lxu = cfg.Lxu;
    env.Ltot = sum(cfg.Lxu);
    env.ranges = user_ranges(cfg.Lxu);
    env.carrier = nrCarrierConfig('NSizeGrid', cfg.carrier.NRB, ...
        'SubcarrierSpacing', cfg.carrier.scs);
    env.ofdm = nrOFDMInfo(env.carrier);
    env.numSubcarriers = 12 * cfg.carrier.NRB;
    env.numSymbols = env.carrier.SymbolsPerSlot;
    env.slotDuration = 1e-3 / (cfg.carrier.scs / 15);
    env.tdd = tdd_pattern(cfg, env);
    env.pcap = 10 ^ (cfg.snrDb / 10) * ones(U, 1);
    env.gain = 10 .^ (cfg.userGainDb(:) / 10);

    groupSize = 12 * cfg.prgRB;
    env.numGroups = env.numSubcarriers / groupSize;

    if env.numGroups ~= fix(env.numGroups)
        error('the carrier must hold a whole number of PRGs');
    end

    env.groups = cell(1, env.numGroups);

    for g = 1:env.numGroups
        env.groups{g} = (g - 1) * groupSize + (1:groupSize);
    end

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

    env.dci = nr_dci(env.catalogs, cfg.Lxu, cfg.carrier.NRB);
    env.dmrsSymbols = numel(env.dmrs.symbols);
    env.srsSymbols = numel(env.srs.symbols);

    if ~isempty(cfg.noise.covariance)
        env.noiseFactor = chol(cfg.noise.covariance, 'lower');
    else
        env.noiseFactor = eye(cfg.numRx);
    end

    env.dmrsBoost = 10 ^ (cfg.dmrs.boostDb / 20);
end

function srs = srs_mapping(cfg, env)
    U = env.U;
    srs = struct();
    srs.indices = cell(1, U);
    srs.values = cell(1, U);
    srs.config = cell(1, U);

    for u = 1:U
        config = nrSRSConfig;
        config.NumSRSPorts = cfg.Lxu(u);
        config.NumSRSSymbols = cfg.srs.symbols;
        config.SymbolStart = cfg.srs.symbolStart;
        config.KTC = cfg.srs.ktc;
        config.CSRS = cfg.srs.csrs;
        config.CyclicShift = cfg.srs.cyclicShifts(u);
        config.SRSPeriod = [1 0];
        carrier = env.carrier;
        carrier.NSlot = 0;
        indices = double(nrSRSIndices(carrier, config, 'IndexStyle', 'subscript', ...
            'IndexBase', '1based'));
        values = unit_power(nrSRS(carrier, config));
        perPort = size(values, 1);
        srs.indices{u} = reshape(indices(:, 1:2), perPort, cfg.Lxu(u), 2);
        srs.values{u} = values;
        srs.config{u} = config;
    end

    positions = squeeze(srs.indices{1}(:, 1, :));

    for u = 2:U

        if ~isequal(squeeze(srs.indices{u}(:, 1, :)), positions)
            error('every UE must sound on the same comb for a joint observation');
        end

    end

    srs.positions = positions;
    srs.symbols = unique(positions(:, 2))';
    srs.groupOf = group_of(positions(:, 1), env);
    srs.portEnergy = env.pcap(:) * cfg.srs.ktc ./ cfg.Lxu(:);
end

function dmrs = dmrs_mapping(cfg, env)
    U = env.U;
    dmrs = struct();
    dmrs.values = cell(U, max(cfg.Lxu));
    dmrs.positions = [];

    for u = 1:U

        for rank = 1:cfg.Lxu(u)
            pusch = nrPUSCHConfig;
            pusch.NumLayers = rank;
            pusch.NumAntennaPorts = rank;
            pusch.PRBSet = 0:cfg.carrier.NRB - 1;
            pusch.TransmissionScheme = 'nonCodebook';
            pusch.DMRS.DMRSConfigurationType = 1;
            pusch.DMRS.DMRSAdditionalPosition = cfg.dmrs.additionalPosition;
            pusch.DMRS.DMRSLength = cfg.dmrs.length;
            pusch.DMRS.NumCDMGroupsWithoutData = 2;
            pusch.DMRS.DMRSPortSet = cfg.dmrs.ports{u}(1:rank);
            values = unit_power(nrPUSCHDMRS(env.carrier, pusch));
            indices = double(nrPUSCHDMRSIndices(env.carrier, pusch, ...
                'IndexStyle', 'subscript', 'IndexBase', '1based'));
            perLayer = size(values, 1);
            positions = indices(1:perLayer, 1:2);

            for layer = 2:rank

                if ~isequal(indices((layer - 1) * perLayer + (1:perLayer), 1:2), positions)
                    error('the DMRS layers of one UE must share a CDM group');
                end

            end

            if isempty(dmrs.positions)
                dmrs.positions = positions;
            elseif ~isequal(dmrs.positions, positions)
                error('the DMRS ports of every UE must share one CDM group');
            end

            dmrs.values{u, rank} = values;
        end

    end

    dmrs.symbols = unique(dmrs.positions(:, 2))';
    dmrs.groupOf = group_of(dmrs.positions(:, 1), env);
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

    if any(pattern == 'S') && cfg.csirs.symbolStart + cfg.csirs.symbols > cfg.tdd.special(1)
        error('nr_setup:tdd', 'the CSI-RS symbols must fit in the downlink part of the S slot');
    end

    for slot = cfg.srs.offset:cfg.srs.periodSlots:cfg.srs.offset + lcm(numel(pattern), ...
            cfg.srs.periodSlots)
        type = pattern(mod(slot, numel(pattern)) + 1);

        if type == 'D' || (type == 'S' && cfg.srs.symbolStart < env.numSymbols - cfg.tdd.special(3))
            error('nr_setup:tdd', 'SRS slot %d has no uplink symbols for the SRS', slot);
        end

    end

end

function csirs = csirs_mapping(cfg, env)
    %CSIRS_MAPPING NZP CSI-RS resources of the reciprocity loop, one per SIC term of a UE
    %
    % Each resource has one port per gNB antenna, density one, and FD-CDM2,
    % and the resources of a UE sit at distinct subcarrier offsets of the
    % same symbols. The noise gain of the UE's estimator is measured once on
    % pure noise, which a UE knows of its own receiver
    rows = containers.Map({2, 4}, {3, 4});
    spacing = containers.Map({2, 4}, {2, 4});

    if ~isKey(rows, cfg.numRx)
        error('nr_setup:csirs', 'CSI-RS needs 2 or 4 gNB antennas, not %d', cfg.numRx);
    end

    symbols = cfg.csirs.symbolStart + (0:cfg.csirs.symbols - 1);
    count = numel(symbols);
    offsets = 0:spacing(cfg.numRx):11;
    csirs = struct('resources', {cell(1, env.U)}, 'imRE', 4 * cfg.carrier.NRB);
    stream = RandStream('mt19937ar', 'Seed', 4093);

    for r = 1:env.U
        config = nrCSIRSConfig('CSIRSType', repmat({'nzp'}, 1, count), ...
            'CSIRSPeriod', 'on', 'RowNumber', rows(cfg.numRx) * ones(1, count), ...
            'Density', repmat({'one'}, 1, count), 'SymbolLocations', num2cell(symbols), ...
            'SubcarrierLocations', repmat({offsets(mod(r - 1, numel(offsets)) + 1)}, 1, count), ...
            'NumRB', cfg.carrier.NRB);
        carrier = env.carrier;
        carrier.NSlot = 0;
        linear = double(nrCSIRSIndices(carrier, config));
        subscripts = double(nrCSIRSIndices(carrier, config, 'IndexStyle', 'subscript', ...
            'IndexBase', '1based'));
        values = nrCSIRS(carrier, config);
        values = values ./ abs(values);
        [cells, ~, slotRE] = unique(sub2ind([env.numSubcarriers, env.numSymbols], ...
            subscripts(:, 1), subscripts(:, 2)));
        portSymbols = zeros(numel(cells), cfg.numRx);
        portSymbols(sub2ind(size(portSymbols), slotRE, subscripts(:, 3))) = values;
        [subcarriers, ~] = ind2sub([env.numSubcarriers, env.numSymbols], cells);
        resource = struct('linear', linear, 'values', values, 'cells', cells, ...
            'portSymbols', portSymbols, 'cellGroup', group_of(subcarriers, env));
        resource.reads = cell(1, cfg.numRx);

        for p = 1:cfg.numRx
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
        resource.numRE = numel(cells);
        csirs.resources{r} = resource;
    end

end

function gain = noise_gain(env, cfg, resource, stream)
    %NOISE_GAIN Output variance of the UE's CSI-RS estimator per unit input noise
    trials = 20;
    total = 0;
    entries = 0;

    for i = 1:trials
        grid = (randn(stream, env.numSubcarriers, env.numSymbols) + ...
            1j * randn(stream, env.numSubcarriers, env.numSymbols)) / sqrt(2);
        estimate = nrChannelEstimate(env.carrier, grid, resource.linear, resource.values, ...
            'CDMLengths', [2 1], 'AveragingWindow', cfg.csirs.averaging, ...
            'PRGBundleSize', cfg.prgRB, 'Interpolation', 'off');

        for p = 1:cfg.numRx
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
    groupSize = env.numSubcarriers / env.numGroups;
    groupOf = ceil(subcarriers / groupSize);
end

function ranges = user_ranges(Lxu)
    ends = cumsum(Lxu);
    starts = [1, ends(1:end - 1) + 1];
    ranges = cell(1, numel(Lxu));

    for u = 1:numel(Lxu)
        ranges{u} = starts(u):ends(u);
    end

end
