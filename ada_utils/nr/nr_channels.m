function chan = nr_channels(cfg, env, seed)
    %NR_CHANNELS Continuously evolving TDL channels of every UE for one realization
    %
    % Each UE gets its own nrTDLChannel whose fading process runs without
    % interruption from slot to slot, so mobility is the toolbox's Doppler
    % evolution rather than a regenerated or autoregressive approximation
    %
    % Args:
    %   cfg: Configuration from nr_default_config
    %   env: Environment from nr_setup
    %   seed: Realization seed
    %
    % Returns:
    %   chan: Channel objects, path filters, and the per-UE amplitude scaling

    U = env.U;
    chan = struct();
    chan.objects = cell(1, U);
    chan.filters = cell(1, U);
    chan.scale = sqrt(env.gain(:));
    chan.offsets = nan(1, U);
    speeds = cfg.channel.speedKmh(:) .* ones(U, 1);
    samplesPerSlot = round(env.ofdm.SampleRate * env.slotDuration);

    for u = 1:U
        doppler = speeds(u) / 3.6 * cfg.channel.carrierGHz * 1e9 / physconst('LightSpeed');
        tdl = nrTDLChannel;
        tdl.DelayProfile = cfg.channel.delayProfile;
        tdl.DelaySpread = cfg.channel.delaySpread;
        tdl.MaximumDopplerShift = doppler;
        tdl.NumTransmitAntennas = cfg.Lxu(u);
        tdl.NumReceiveAntennas = cfg.numRx;
        tdl.MIMOCorrelation = get_opt(cfg.channel, 'correlation', 'Low');

        if strcmp(tdl.MIMOCorrelation, 'Medium') || strcmp(tdl.MIMOCorrelation, 'High')
            tdl.TransmissionDirection = 'Uplink';
        end

        tdl.SampleRate = env.ofdm.SampleRate;
        tdl.ChannelFiltering = false;
        tdl.NormalizeChannelOutputs = false;
        tdl.NumTimeSamples = samplesPerSlot;
        tdl.RandomStream = 'mt19937ar with seed';
        tdl.Seed = seed * 16 + u;
        chan.objects{u} = tdl;
        chan.filters{u} = getPathFilters(tdl);
        chan.doppler(u) = doppler;
    end

end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
