function chan = nr_channels(cfg, env, seed)
    %NR_CHANNELS Continuously evolving CDL channels from every UE array to the gNB panel
    %
    % Each UE gets its own nrCDLChannel whose fading process runs without
    % interruption, so mobility is the toolbox's Doppler evolution. The CDL
    % profile is set up in downlink terms (gNB transmits) with the UE at a
    % random azimuth inside the sector and a random array orientation, then
    % swapped to the reciprocal uplink link
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %   env: Environment from nr_setup
    %   seed: Realization seed
    %
    % Returns:
    %   chan: Channel objects, path filters, and the per-UE amplitude scaling

    U = env.U;
    stream = RandStream('mt19937ar', 'Seed', seed);
    speeds = cfg.channel.speedKmh(:) .* ones(U, 1);
    chan = struct('objects', {cell(1, U)}, 'filters', {cell(1, U)}, ...
        'scale', sqrt(env.gain(:)), 'offsets', nan(1, U));

    for u = 1:U
        cdl = nrCDLChannel;
        cdl.DelayProfile = cfg.channel.delayProfile;
        cdl.DelaySpread = cfg.channel.delaySpread;
        cdl.CarrierFrequency = cfg.channel.carrierGHz * 1e9;
        cdl.MaximumDopplerShift = speeds(u) / 3.6 * cdl.CarrierFrequency / physconst('LightSpeed');
        cdl.TransmitAntennaArray.Size = [cfg.gnbArray, 1, 1];
        cdl.ReceiveAntennaArray.Size = [ue_array(cfg.Lxu(u)), 1, 1];
        cdl.AngleScaling = true;
        cdl.AngleSpreads = cfg.channel.angleSpreads;
        cdl.MeanAngles = [cfg.channel.sectorDeg * (rand(stream) - 0.5), ...
            360 * (rand(stream) - 0.5), 90, 90];
        cdl.SampleRate = env.ofdm.SampleRate;
        cdl.ChannelFiltering = false;
        cdl.NormalizeChannelOutputs = false;
        cdl.NumTimeSamples = round(env.ofdm.SampleRate * env.slotDuration);
        cdl.RandomStream = 'mt19937ar with seed';
        cdl.Seed = seed * 16 + u;
        swapTransmitAndReceive(cdl);
        chan.objects{u} = cdl;
        chan.filters{u} = getPathFilters(cdl);
    end

end

function shape = ue_array(ports)
    %UE_ARRAY Rows, columns, and polarizations of a UE array with this many ports

    switch ports
        case 1
            shape = [1 1 1];
        case 2
            shape = [1 1 2];
        case 4
            shape = [1 2 2];
        otherwise
            error('nr_channels:ports', 'UEs have 1, 2, or 4 antenna ports, not %d', ports);
    end

end
