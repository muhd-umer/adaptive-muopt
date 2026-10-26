function cfg = nr_default_config()
    %NR_DEFAULT_CONFIG Physical, signaling, and evaluation settings shared by the NR experiments
    %
    % Energies are per resource element and normalized to unit noise per
    % receive antenna, so pcap = 10^(snrDb / 10) is the full-power energy a UE
    % spreads over its ports and userGainDb is its large-scale gain
    %
    % Returns:
    %   cfg: Configuration struct

    cfg = struct();
    cfg.numRx = 4;
    cfg.Lxu = [2 2];
    cfg.theta = [1; 1.5];
    cfg.cb = 1;
    cfg.snrDb = 5;
    cfg.userGainDb = [0 -3];

    % DDDSU at 30 kHz is the common 2.5 ms pattern of the 3.5 GHz band;
    % special holds the downlink, guard, and uplink symbols of the S slot
    cfg.carrier = struct('NRB', 24, 'scs', 30);
    cfg.tdd = struct('pattern', 'DDDSU', 'special', [10 2 2]);
    cfg.prgRB = 2;
    cfg.evalSymbol = 7;
    cfg.selectStride = 8;

    % with uncorrelated UE antennas (TS 38.104 uplink low) the identity
    % precoder is nearly optimal, so the medium correlation is the default
    cfg.channel = struct('delayProfile', 'TDL-C', 'delaySpread', 100e-9, ...
        'carrierGHz', 3.5, 'speedKmh', [15 15], 'correlation', 'Medium');
    cfg.noise = struct('covariance', []);

    % SRS sits in the uplink symbols of the S slot, on one comb shared by both
    % UEs through cyclic shifts; ports 0-1 and 4-5 share DMRS CDM group 0
    % through double-symbol DMRS, so the precoded layers are jointly observed
    cfg.srs = struct('periodSlots', 5, 'offset', 3, 'symbols', 2, ...
        'symbolStart', 12, 'ktc', 2, 'csrs', 6, 'cyclicShifts', [0 2]);
    cfg.dmrs = struct('additionalPosition', 1, 'length', 2, ...
        'ports', {{[0 1], [4 5]}}, 'boostDb', 3);

    cfg.codebook = struct('codebookSubset', 'fullyAndPartialAndNonCoherent', ...
        'powerMode', 'fullpower', 'maxRank', []);
    cfg.power = struct('floorDb', -20);
    cfg.control = struct('delay', 2, 'pDrop', 0, 'pLate', 0, 'lateSlots', 1);
    cfg.link = struct('ollaStep', 0.05, 'targetBler', 0.1);

    cfg.policy = struct('triggerRelative', 0.01, 'dmrsForget', 0.6, 'srsForget', 1, ...
        'maxSweeps', 3);
    cfg.modeB = struct('energyBits', 4, 'shareBits', 3, 'angleBits', 3, ...
        'phaseBits', 4, 'energyRangeDb', 30, 'deltaFraction', 0.25, ...
        'floorFraction', 0.1, 'stepFraction', 0.2, 'triggerRelative', 5e-3, ...
        'innerSteps', 6, 'bundle', 1);
    cfg.ue = struct('iterations', 40, 'budgetPeriod', 2, 'budgetStep', 0.3);

    % the gNB sends its precoded CSI-RS in the first downlink or special slot
    % at least lagSlots after the uplink DMRS; calibration is the rms residual
    % per-antenna mismatch of the transmit and receive chains
    cfg.csirs = struct('snrOffsetDb', 10, 'symbolStart', 5, 'symbols', 1, ...
        'averaging', [1 1], 'lagSlots', 1, ...
        'calibration', struct('amplitudeDb', 0.5, 'phaseDeg', 5));

    cfg.horizonSlots = 400;
    cfg.warmupSlots = 100;
    cfg.realizations = 100;
    cfg.seed = 2026;
    cfg.parallel = true;
    cfg.activity = [];
    cfg.methods = {'canonical', 'nr-oracle', 'nr-csi', 'nr-pilot', 'cov-pilot', 'hold'};
end
