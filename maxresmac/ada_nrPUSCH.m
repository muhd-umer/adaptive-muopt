clear;
clc;

cfg.Nr = 4;
cfg.Nt = [2 2];
cfg.theta = [1; 1.5];
cfg.cb = 1;
cfg.scs = 30;
cfg.NRB = 24;
cfg.numPRG = 12;
cfg.snrDb = 10;
cfg.userGainDb = [0 -3];
cfg.realizations = 4;
cfg.seed = 2026;
cfg.delayProfile = 'TDL-C';
cfg.delaySpread = 100e-9;
cfg.carrierFrequencyGHz = 3.5;
cfg.speedKmh = 60;
cfg.doppler = cfg.speedKmh / 3.6 * cfg.carrierFrequencyGHz * 1e9 / 299792458;
cfg.totalSlots = 100;
cfg.trackingPeriodSlots = 2;
cfg.canonicalPeriodSlots = 20;
cfg.srsSymbols = 2;
cfg.srsComb = 2;
cfg.srsPeriodSlots = cfg.canonicalPeriodSlots;
cfg.trackingPilots = 24;
cfg.feedbackThreshold = 0.015;
cfg.fullFeedbackBitsPerUserPRG = 32;
cfg.deltaFeedbackBitsPerUserPRG = 8;
cfg.codebookFeedbackBitsPerUserPRG = 4;

Lxu = cfg.Nt;
Ltot = sum(Lxu);
U = numel(Lxu);
theta = cfg.theta(:);
Etotal = Ltot * cfg.numPRG;
noiseVar = 1;
userGain = 10 .^ (cfg.userGainDb / 10);
channelScale = sqrt(10 ^ (cfg.snrDb / 10) / sum(Lxu .* userGain));
priorVariance = channelScale ^ 2 * repelem(userGain, Lxu);
use5g = exist('nrTDLChannel', 'class') == 8 && exist('nrSRS', 'file') == 2;
[srsSamples, srsDescription] = srs_budget(cfg, use5g);
numIntervals = cfg.totalSlots / cfg.trackingPeriodSlots;
anchorStride = cfg.canonicalPeriodSlots / cfg.trackingPeriodSlots;
slotDuration = 1e-3 / (cfg.scs / 15);
intervalDuration = cfg.trackingPeriodSlots * slotDuration;
rho = besselj(0, 2 * pi * cfg.doppler * intervalDuration);
resourcesPerPRGInterval = 24 * 14 * cfg.trackingPeriodSlots;
fullFeedbackBits = U * cfg.numPRG * cfg.fullFeedbackBitsPerUserPRG;
deltaFeedbackBits = U * cfg.numPRG * cfg.deltaFeedbackBitsPerUserPRG;
upperLabel = sprintf('perfect CSI every %d slots', cfg.trackingPeriodSlots);

if mod(cfg.totalSlots, cfg.trackingPeriodSlots) ~= 0 || ...
        mod(cfg.canonicalPeriodSlots, cfg.trackingPeriodSlots) ~= 0
    error('slot periods must be integer multiples');
end

upperGross = zeros(cfg.realizations, numIntervals);
conventionalGross = zeros(cfg.realizations, numIntervals);
standardGross = zeros(cfg.realizations, numIntervals);
slowGross = zeros(cfg.realizations, numIntervals);
hybridGross = zeros(cfg.realizations, numIntervals);
conventionalNet = zeros(cfg.realizations, numIntervals);
standardNet = zeros(cfg.realizations, numIntervals);
slowNet = zeros(cfg.realizations, numIntervals);
hybridNet = zeros(cfg.realizations, numIntervals);
conventionalOverhead = zeros(cfg.realizations, numIntervals);
slowOverhead = zeros(cfg.realizations, numIntervals);
hybridOverhead = zeros(cfg.realizations, numIntervals);
estimateNmse = nan(cfg.realizations, numIntervals);
upperMs = zeros(cfg.realizations, numIntervals);
conventionalMs = zeros(cfg.realizations, numIntervals);
standardMs = zeros(cfg.realizations, numIntervals);
slowMs = zeros(cfg.realizations, numIntervals);
hybridMs = zeros(cfg.realizations, numIntervals);
hybridEvaluations = zeros(cfg.realizations, numIntervals);
slowUpdates = zeros(cfg.realizations, 1);
hybridFullUpdates = zeros(cfg.realizations, 1);
hybridDeltaUpdates = zeros(cfg.realizations, 1);

fprintf('[two-timescale 5g nr uplink mac]\n');
fprintf('  BS antennas=%d, users=%d, UE antennas=[%s], weights=[%s]\n', ...
    cfg.Nr, U, num2str(Lxu), num2str(theta'));
fprintf('  channel=%s, delay spread=%.0f ns, Doppler=%g Hz, correlation=%.4f/track\n', ...
    ternary(use5g, '5G Toolbox TDL-C', 'self-contained TDL-C-like fallback'), ...
    cfg.delaySpread * 1e9, cfg.doppler, rho);
fprintf('  mobility=%.0f km/h at %.1f GHz\n', ...
    cfg.speedKmh, cfg.carrierFrequencyGHz);
fprintf('  carrier=%d RB, %d kHz SCS, %d two-RB PRGs, SNR=%g dB\n', ...
    cfg.NRB, cfg.scs, cfg.numPRG, cfg.snrDb);
fprintf('  canonical anchor=%d slots, adaptive tracking=%d slots, horizon=%d slots\n', ...
    cfg.canonicalPeriodSlots, cfg.trackingPeriodSlots, cfg.totalSlots);
fprintf('  anchor SRS=%s, %d RE/port/PRG\n', srsDescription, srsSamples);
fprintf('  adaptive observations=%d scheduled DMRS RE/PRG, feedback trigger=%.1f%%\n\n', ...
    cfg.trackingPilots, 100 * cfg.feedbackThreshold);

for realization = 1:cfg.realizations
    rng(cfg.seed + realization);
    Hfull = generate_nr_channel(cfg, Lxu, userGain, use5g);
    Htrajectory = cell(1, numIntervals);
    Htrajectory{1} = channelScale * select_prg_channels(Hfull, cfg.numPRG);

    for interval = 2:numIntervals
        innovation = tracking_innovation(cfg, Lxu, userGain, use5g);
        Htrajectory{interval} = rho * Htrajectory{interval - 1} + ...
            sqrt(max(1 - rho ^ 2, 0)) * channelScale * innovation;
    end

    Rslow = equal_power_design(Lxu, cfg.numPRG, Etotal);
    Rhybrid = Rslow;

    for interval = 1:numIntervals
        H = Htrajectory{interval};

        tic;
        [Rupper, ~, ~, ~] = maxRESMACMIMO(H, Lxu, Etotal, theta, cfg.cb, false);
        upperMs(realization, interval) = toc * 1000;
        upperGross(realization, interval) = weighted_rate(H, Rupper, Lxu, theta, cfg.cb);
        isAnchor = mod(interval - 1, anchorStride) == 0;
        estimateSeed = cfg.seed + 100000 * realization + interval;
        [Hestimate, estimateNmse(realization, interval)] = estimate_from_srs( ...
            H, srsSamples, noiseVar, 1, priorVariance, estimateSeed);
        tic;
        [Rconventional, ~, ~, ~] = maxRESMACMIMO( ...
            Hestimate, Lxu, Etotal, theta, cfg.cb, false);
        conventionalMs(realization, interval) = toc * 1000;
        conventionalGross(realization, interval) = weighted_rate( ...
            H, Rconventional, Lxu, theta, cfg.cb);
        conventionalOverhead(realization, interval) = ...
            srsSamples / resourcesPerPRGInterval;
        conventionalNet(realization, interval) = conventionalGross(realization, interval) * ...
            (1 - conventionalOverhead(realization, interval));
        tic;
        Rstandard = nr_codebook_design( ...
            Hestimate, Lxu, Etotal, theta, cfg.cb, use5g);
        standardMs(realization, interval) = toc * 1000;
        standardGross(realization, interval) = weighted_rate( ...
            H, Rstandard, Lxu, theta, cfg.cb);
        standardNet(realization, interval) = standardGross(realization, interval) * ...
            (1 - conventionalOverhead(realization, interval));

        if isAnchor
            Ranchor = Rconventional;
            elapsed = conventionalMs(realization, interval);
            Rslow = Ranchor;
            Rhybrid = Ranchor;
            slowMs(realization, interval) = elapsed;
            hybridMs(realization, interval) = elapsed;
            slowUpdates(realization) = slowUpdates(realization) + 1;
            hybridFullUpdates(realization) = hybridFullUpdates(realization) + 1;
            overhead = srsSamples / resourcesPerPRGInterval;
            slowOverhead(realization, interval) = overhead;
            hybridOverhead(realization, interval) = overhead;
        else
            pilot = struct();
            pilot.numRx = cfg.Nr;
            pilot.numTones = cfg.numPRG;
            pilot.sample = @(n, Atx, X) receive_pilots( ...
                H(:, :, n), Atx, X, noiseVar);
            opts = struct();
            opts.R0 = Rhybrid;
            opts.numPilots = cfg.trackingPilots;
            opts.singlePass = true;
            opts.stepInitial = 0.5;
            opts.floorInitial = 1e-5;
            opts.floorFinal = 1e-7;
            opts.rngSeed = cfg.seed + 200000 * realization + interval;
            tic;
            [Rcandidate, ~, ~, ~, info] = ada_maxRESMACMIMO( ...
                pilot, Lxu, Etotal, theta, cfg.cb, opts);
            hybridMs(realization, interval) = toc * 1000;
            hybridEvaluations(realization, interval) = info.pilotCalls / cfg.numPRG;
            hybridOverhead(realization, interval) = 0;

            deployUpdate = covariance_change(Rcandidate, Rhybrid) >= ...
                cfg.feedbackThreshold;

            if deployUpdate
                Rhybrid = Rcandidate;
                hybridDeltaUpdates(realization) = hybridDeltaUpdates(realization) + 1;
            end
        end

        slowGross(realization, interval) = weighted_rate( ...
            H, Rslow, Lxu, theta, cfg.cb);
        hybridGross(realization, interval) = weighted_rate( ...
            H, Rhybrid, Lxu, theta, cfg.cb);
        slowNet(realization, interval) = slowGross(realization, interval) * ...
            (1 - slowOverhead(realization, interval));
        hybridNet(realization, interval) = hybridGross(realization, interval) * ...
            (1 - hybridOverhead(realization, interval));
    end

    fprintf('  realization %d/%d data rate: upper %.3f, NR codebook %.3f, SRS-canonical %.3f, held %.3f, adaptive %.3f\n', ...
        realization, cfg.realizations, mean(upperGross(realization, :)), ...
        mean(standardNet(realization, :)), ...
        mean(conventionalNet(realization, :)), ...
        mean(slowNet(realization, :)), mean(hybridNet(realization, :)));
end

durationSeconds = cfg.totalSlots * slotDuration;
slowBits = slowUpdates * fullFeedbackBits;
hybridBits = hybridFullUpdates * fullFeedbackBits + ...
    hybridDeltaUpdates * deltaFeedbackBits;
conventionalBits = numIntervals * fullFeedbackBits;
standardBits = numIntervals * U * cfg.numPRG * ...
    cfg.codebookFeedbackBitsPerUserPRG;
upperHypotheticalBits = numIntervals * fullFeedbackBits;
upper = mean(upperGross, 'all');

fprintf('\n[data rate after incremental design pilots]\n');
fprintf('%-28s %14s %12s %12s\n', ...
    'method', 'data rate', 'gap', 'incr pilot');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-28s %14.4f %11.2f%% %11.2f%%\n', ...
    upperLabel, upper, 0, 0);
fprintf('%-28s %14.4f %11.2f%% %11.2f%%\n', ...
    'NR SRS + RI/TPMI codebook', mean(standardNet, 'all'), ...
    100 * relative_gap(mean(standardNet, 'all'), upper), ...
    100 * mean(conventionalOverhead, 'all'));
fprintf('%-28s %14.4f %11.2f%% %11.2f%%\n', ...
    'frequent SRS + canonical', mean(conventionalNet, 'all'), ...
    100 * relative_gap(mean(conventionalNet, 'all'), upper), ...
    100 * mean(conventionalOverhead, 'all'));
fprintf('%-28s %14.4f %11.2f%% %11.2f%%\n', ...
    '20-slot canonical hold', mean(slowNet, 'all'), ...
    100 * relative_gap(mean(slowNet, 'all'), upper), ...
    100 * mean(slowOverhead, 'all'));
fprintf('%-28s %14.4f %11.2f%% %11.2f%%\n', ...
    'anchors + adaptive tracking', mean(hybridNet, 'all'), ...
    100 * relative_gap(mean(hybridNet, 'all'), upper), ...
    100 * mean(hybridOverhead, 'all'));

fprintf('\n[feedback over %.1f ms]\n', 1000 * durationSeconds);
fprintf('%-28s %12s %12s %14s\n', 'method', 'design updates', ...
    'delta updates', 'bits / kbit/s');
fprintf('%s\n', repmat('-', 1, 73));
fprintf('%-28s %12d %12d %7d / %6.1f\n', ...
    upperLabel, numIntervals, 0, upperHypotheticalBits, ...
    upperHypotheticalBits / durationSeconds / 1000);
fprintf('%-28s %12d %12d %7d / %6.1f\n', ...
    'NR SRS + RI/TPMI codebook', numIntervals, 0, standardBits, ...
    standardBits / durationSeconds / 1000);
fprintf('%-28s %12d %12d %7d / %6.1f\n', ...
    'frequent SRS + canonical', numIntervals, 0, conventionalBits, ...
    conventionalBits / durationSeconds / 1000);
fprintf('%-28s %12.1f %12d %7.0f / %6.1f\n', ...
    '20-slot canonical hold', mean(slowUpdates), 0, mean(slowBits), ...
    mean(slowBits) / durationSeconds / 1000);
fprintf('%-28s %12.1f %12.1f %7.0f / %6.1f\n', ...
    'anchors + adaptive tracking', mean(hybridFullUpdates), ...
    mean(hybridDeltaUpdates), mean(hybridBits), ...
    mean(hybridBits) / durationSeconds / 1000);
fprintf('  feedback model: %d bits/full and %d bits/delta per user/PRG\n', ...
    cfg.fullFeedbackBitsPerUserPRG, cfg.deltaFeedbackBitsPerUserPRG);
fprintf('  NR codebook feedback model: %d RI/TPMI bits per user/PRG\n', ...
    cfg.codebookFeedbackBitsPerUserPRG);
fprintf('  scheduled PUSCH DMRS is common receiver overhead and is not charged twice\n');

fprintf('\n[computation and estimation]\n');
fprintf('  perfect-CSI canonical solve:       %8.2f ms/interval\n', mean(upperMs, 'all'));
fprintf('  estimated-CSI canonical solve:     %8.2f ms/interval\n', ...
    mean(conventionalMs, 'all'));
fprintf('  NR RI/TPMI codebook selection:     %8.2f ms/interval\n', ...
    mean(standardMs, 'all'));
fprintf('  periodic canonical anchor:         %8.2f ms/anchor\n', ...
    sum(slowMs, 'all') / nnz(slowMs));
fprintf('  adaptive tracking solve:           %8.2f ms/tracking update\n', ...
    sum(hybridMs(~slowMs)) / nnz(~slowMs));
fprintf('  adaptive design evaluations:       %8.2f/update\n', ...
    sum(hybridEvaluations, 'all') / nnz(hybridEvaluations));
fprintf('  canonical-hold average compute:     %8.2f ms/interval\n', ...
    sum(slowMs, 'all') / numel(slowMs));
fprintf('  adaptive-tracking average compute:  %8.2f ms/interval\n', ...
    sum(hybridMs, 'all') / numel(hybridMs));
fprintf('  conventional SRS estimate NMSE:    %8.2f dB\n', ...
    10 * log10(mean(estimateNmse(~isnan(estimateNmse)))));

function [samplesPerPRG, description] = srs_budget(cfg, use5g)
    if use5g
        carrier = nrCarrierConfig;
        carrier.NSizeGrid = cfg.NRB;
        carrier.SubcarrierSpacing = cfg.scs;
        srs = nrSRSConfig;
        srs.NumSRSPorts = max(cfg.Nt);
        srs.NumSRSSymbols = cfg.srsSymbols;
        srs.SymbolStart = 14 - cfg.srsSymbols;
        srs.KTC = cfg.srsComb;
        srs.CSRS = 6;
        srs.SRSPeriod = [cfg.srsPeriodSlots 0];
        symbols = nrSRS(carrier, srs);
        samplesPerPRG = floor(size(symbols, 1) / cfg.numPRG);
    else
        subcarriers = cfg.NRB * 12 / cfg.numPRG;
        samplesPerPRG = floor(subcarriers / cfg.srsComb) * cfg.srsSymbols;
    end

    description = sprintf( ...
        '%d ports/UE, %d symbols, comb-%d, period %d slots', ...
        max(cfg.Nt), cfg.srsSymbols, cfg.srsComb, cfg.srsPeriodSlots);
end

function H = generate_nr_channel(cfg, Lxu, userGain, use5g)
    numSubcarriers = cfg.NRB * 12;
    H = zeros(cfg.Nr, sum(Lxu), numSubcarriers);
    offset = 0;

    for user = 1:numel(Lxu)
        if use5g
            Hu = nr_tdl_frequency_response(cfg, Lxu(user), numSubcarriers);
        else
            Hu = fallback_tdl_response(cfg, Lxu(user), numSubcarriers);
        end

        Hu = Hu / sqrt(mean(abs(Hu(:)) .^ 2));
        columns = offset + (1:Lxu(user));
        H(:, columns, :) = sqrt(userGain(user)) * Hu;
        offset = offset + Lxu(user);
    end
end

function innovation = tracking_innovation(cfg, Lxu, userGain, use5g)
    full = generate_nr_channel(cfg, Lxu, userGain, use5g);
    innovation = select_prg_channels(full, cfg.numPRG);
end

function Hu = nr_tdl_frequency_response(cfg, Nt, numSubcarriers)
    ofdmInfo = nrOFDMInfo(cfg.NRB, cfg.scs);
    channel = nrTDLChannel;
    channel.DelayProfile = cfg.delayProfile;
    channel.DelaySpread = cfg.delaySpread;
    channel.MaximumDopplerShift = cfg.doppler;
    channel.NumTransmitAntennas = Nt;
    channel.NumReceiveAntennas = cfg.Nr;
    channel.SampleRate = ofdmInfo.SampleRate;
    channel.RandomStream = 'Global stream';
    numSamples = round(ofdmInfo.SampleRate * 1e-3);
    input = (randn(numSamples, Nt) + 1j * randn(numSamples, Nt)) / sqrt(2);
    [~, pathGains] = channel(input);
    pathFilters = getPathFilters(channel);
    grid = nrPerfectChannelEstimate(pathGains, pathFilters, cfg.NRB, cfg.scs, 0);
    symbol = max(1, ceil(size(grid, 2) / 2));
    Hu = zeros(cfg.Nr, Nt, numSubcarriers);

    for subcarrier = 1:numSubcarriers
        Hu(:, :, subcarrier) = squeeze(grid(subcarrier, symbol, :, :));
    end
end

function Hu = fallback_tdl_response(cfg, Nt, numSubcarriers)
    numTaps = 8;
    tapIndex = 0:numTaps - 1;
    profile = exp(-tapIndex / 2);
    profile = profile / sum(profile);
    Hu = zeros(cfg.Nr, Nt, numSubcarriers);

    for receive = 1:cfg.Nr
        for transmit = 1:Nt
            gains = (randn(1, numTaps) + 1j * randn(1, numTaps)) / sqrt(2);
            gains = gains .* sqrt(profile);

            for subcarrier = 1:numSubcarriers
                phase = exp(-2j * pi * (subcarrier - 1) * tapIndex / numSubcarriers);
                Hu(receive, transmit, subcarrier) = sum(gains .* phase);
            end
        end
    end
end

function Hprg = select_prg_channels(H, numPRG)
    numSubcarriers = size(H, 3);
    edges = round(linspace(0, numSubcarriers, numPRG + 1));
    Hprg = zeros(size(H, 1), size(H, 2), numPRG);

    for prg = 1:numPRG
        indices = edges(prg) + 1:edges(prg + 1);
        Hprg(:, :, prg) = H(:, :, indices(ceil(numel(indices) / 2)));
    end
end

function [Hestimate, nmse] = estimate_from_srs( ...
        H, numPilots, noiseVar, pilotPower, priorVariance, seed)
    rng(seed);
    [Nr, Ltot, numPRG] = size(H);
    phases = exp(-2j * pi * (0:Ltot - 1)' * (0:numPilots - 1) / numPilots);
    X = sqrt(pilotPower) * phases;
    Hestimate = zeros(size(H));
    squaredError = 0;
    channelEnergy = 0;

    for prg = 1:numPRG
        noise = sqrt(noiseVar / 2) * ...
            (randn(Nr, numPilots) + 1j * randn(Nr, numPilots));
        Y = H(:, :, prg) * X + noise;
        Hls = Y * X' / (numPilots * pilotPower);

        for transmit = 1:Ltot
            prior = priorVariance(transmit);
            shrinkage = prior * numPilots * pilotPower / ...
                (prior * numPilots * pilotPower + noiseVar);
            Hestimate(:, transmit, prg) = shrinkage * Hls(:, transmit);
        end

        difference = Hestimate(:, :, prg) - H(:, :, prg);
        squaredError = squaredError + norm(difference, 'fro') ^ 2;
        channelEnergy = channelEnergy + norm(H(:, :, prg), 'fro') ^ 2;
    end

    nmse = squaredError / max(channelEnergy, realmin);
end

function Y = receive_pilots(H, Atx, X, noiseVar)
    noise = sqrt(noiseVar / 2) * ...
        (randn(size(H, 1), size(X, 2)) + 1j * randn(size(H, 1), size(X, 2)));
    Y = H * Atx * X + noise;
end

function R = nr_codebook_design(H, Lxu, Etotal, theta, cb, use5g)
    U = numel(Lxu);
    numPRG = size(H, 3);
    energy = Etotal / (U * numPRG);
    candidates = cell(U, 1);
    counts = zeros(U, 1);

    for user = 1:U
        candidates{user} = codebook_covariances(Lxu(user), use5g);
        counts(user) = numel(candidates{user});
    end

    R = cell(U, numPRG);
    combinations = prod(counts);

    for prg = 1:numPRG
        best = -inf;
        bestBlocks = cell(U, 1);

        for combination = 0:combinations - 1
            code = combination;
            blocks = cell(U, 1);

            for user = 1:U
                index = mod(code, counts(user)) + 1;
                code = floor(code / counts(user));
                blocks{user} = energy * candidates{user}{index};
            end

            objective = weighted_tone_rate( ...
                H(:, :, prg), blocks, Lxu, theta, cb);

            if objective > best
                best = objective;
                bestBlocks = blocks;
            end
        end

        R(:, prg) = bestBlocks;
    end
end

function candidates = codebook_covariances(numPorts, use5g)
    if numPorts ~= 2
        candidates = {eye(numPorts) / numPorts};
        return;
    end

    candidates = cell(7, 1);

    if use5g && exist('nrPUSCHCodebook', 'file') == 2
        for tpmi = 0:5
            W = nrPUSCHCodebook(1, numPorts, tpmi);
            A = transpose(W);
            Q = A * A';
            candidates{tpmi + 1} = Q / real(trace(Q));
        end

        W = nrPUSCHCodebook(2, numPorts, 0);
        A = transpose(W);
        Q = A * A';
        candidates{7} = Q / real(trace(Q));
        return;
    end

    beams = [1 0 1 1 1 1; 0 1 1 -1 1j -1j];

    for index = 1:size(beams, 2)
        beam = beams(:, index) / norm(beams(:, index));
        candidates{index} = beam * beam';
    end

    candidates{7} = eye(2) / 2;
end

function objective = weighted_tone_rate(H, R, Lxu, theta, cb)
    ranges = user_ranges(Lxu);
    [~, order] = sort(theta, 'descend');
    previous = eye(size(H, 1));
    rates = zeros(numel(theta), 1);

    for index = 1:numel(theta)
        user = order(index);
        Hu = H(:, ranges{user});
        current = previous + Hu * R{user} * Hu';
        rates(user) = (logdet_spd(current) - logdet_spd(previous)) / ...
            (cb * log(2));
        previous = current;
    end

    objective = theta(:)' * rates;
end

function R = equal_power_design(Lxu, numPRG, Etotal)
    R = cell(numel(Lxu), numPRG);
    energy = Etotal / (sum(Lxu) * numPRG);

    for user = 1:numel(Lxu)
        for prg = 1:numPRG
            R{user, prg} = energy * eye(Lxu(user));
        end
    end
end

function change = covariance_change(left, right)
    numerator = 0;
    denominator = 0;

    for index = 1:numel(left)
        numerator = numerator + norm(left{index} - right{index}, 'fro') ^ 2;
        denominator = denominator + norm(right{index}, 'fro') ^ 2;
    end

    change = sqrt(numerator / max(denominator, realmin));
end

function objective = weighted_rate(H, Rxx, Lxu, theta, cb)
    theta = theta(:);
    U = numel(theta);
    numPRG = size(H, 3);
    ranges = user_ranges(Lxu);
    [~, order] = sort(theta, 'descend');
    rates = zeros(U, numPRG);

    for prg = 1:numPRG
        previous = eye(size(H, 1));

        for k = 1:U
            user = order(k);
            Hu = H(:, ranges{user}, prg);
            current = previous + Hu * Rxx{user, prg} * Hu';
            rates(user, prg) = (logdet_spd(current) - logdet_spd(previous)) / ...
                (cb * log(2));
            previous = current;
        end
    end

    objective = theta' * sum(rates, 2);
end

function ranges = user_ranges(Lxu)
    ends = cumsum(Lxu);
    starts = [1, ends(1:end - 1) + 1];
    ranges = cell(1, numel(Lxu));

    for user = 1:numel(Lxu)
        ranges{user} = starts(user):ends(user);
    end
end

function value = logdet_spd(matrix)
    factor = chol(0.5 * (matrix + matrix'));
    value = 2 * sum(log(real(diag(factor))));
end

function gap = relative_gap(rate, upperBound)
    gap = max(upperBound - rate, 0) / max(abs(upperBound), 1);
end

function value = ternary(condition, ifTrue, ifFalse)
    if condition
        value = ifTrue;
    else
        value = ifFalse;
    end
end
