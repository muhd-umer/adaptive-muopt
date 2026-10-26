function cfg = nr_default_config()
    %NR_DEFAULT_CONFIG Physical, signaling, and evaluation settings shared by the NR experiments
    %
    % Energies are per resource element and normalized to unit noise per
    % receive antenna, so pcap = 10^(snrDb / 10) is the full-power energy a UE
    % spreads over its ports and userGainDb is its large-scale gain. The gNB
    % panel is [rows, columns, polarizations] with numRx = prod(gnbArray)
    % antennas; Lxu lists each UE's ports (1, 2, or 4), and theta and
    % userGainDb hold one entry per UE
    %
    % Returns:
    %   cfg: Configuration struct

    cfg = struct();
    cfg.gnbArray = [4 4 2];
    cfg.numRx = prod(cfg.gnbArray);
    cfg.Lxu = [4 2];
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

    % CDL channels (TR 38.901) between the gNB panel and each UE array, each
    % UE at its own random azimuth within the sector, with the median angle
    % spreads [ASD ASA ZSD ZSA] of urban macro NLOS in downlink terms
    cfg.channel = struct('delayProfile', 'CDL-C', 'delaySpread', 100e-9, ...
        'carrierGHz', 3.5, 'speedKmh', 15, 'sectorDeg', 120, ...
        'angleSpreads', [26 74 5 18]);
    cfg.noise = struct('covariance', []);

    % SRS sits in the uplink symbols of the S slot; UEs share a comb through
    % cyclic shifts so the sounding stays joint. Double-symbol DMRS gives
    % each CDM group four orthogonal ports, type 1 up to 8 layers in all and
    % type 2 up to 12
    cfg.srs = struct('periodSlots', 5, 'offset', 3, 'symbols', 2, ...
        'symbolStart', 12, 'csrs', 6);
    cfg.dmrs = struct('additionalPosition', 1, 'length', 2);

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
    % at least lagSlots after the uplink DMRS, and the UE needs processingSlots
    % (42 symbols) before a PUSCH can use the precoder it computes from it;
    % calibration is the rms residual per-antenna mismatch of the chains
    cfg.csirs = struct('snrOffsetDb', 10, 'symbolStart', 2, 'averaging', [1 1], ...
        'lagSlots', 2, 'processingSlots', 3, ...
        'calibration', struct('amplitudeDb', 0.5, 'phaseDeg', 5));

    cfg.horizonSlots = 400;
    cfg.warmupSlots = 100;
    cfg.realizations = 100;
    cfg.seed = 2026;
    cfg.parallel = true;
    cfg.activity = [];
    cfg.methods = {'canonical', 'nr-oracle', 'nr-csi', 'nr-pilot', 'ue-pilot', 'hold'};
end
