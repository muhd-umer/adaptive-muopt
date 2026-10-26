function [H, chan] = nr_channel_slot(chan, env, slot)
    %NR_CHANNEL_SLOT Grid response of every UE channel in one slot
    %
    % Args:
    %   chan: Channels from nr_channels
    %   env: Environment from nr_setup
    %   slot: Zero-based slot number, which sets the fading time and the OFDM
    %     symbol timing
    %
    % Returns:
    %   H: numRx-by-Ltot-by-numSubcarriers-by-numSymbols frequency response with
    %     the UE streams stacked and the large-scale gains applied
    %   chan: Channels with the timing offsets fixed on first use

    carrier = env.carrier;
    carrier.NSlot = slot;
    numRx = chan.objects{1}.NumReceiveAntennas;
    H = zeros(numRx, env.Ltot, env.numSubcarriers, env.numSymbols);

    for u = 1:env.U
        % the fading process is a deterministic function of time
        release(chan.objects{u});
        chan.objects{u}.InitialTime = slot * env.slotDuration;
        [gains, times] = chan.objects{u}();

        % timing advance keeps every UE aligned, so the bulk channel-filter
        % delay is removed once and then held, as a gNB timing loop would
        if isnan(chan.offsets(u))
            chan.offsets(u) = nrPerfectTimingEstimate(gains, chan.filters{u});
        end

        grid = nrPerfectChannelEstimate(carrier, gains, chan.filters{u}, ...
            chan.offsets(u), times);
        H(:, env.ranges{u}, :, :) = chan.scale(u) * permute(grid, [3 4 1 2]);
    end

end
