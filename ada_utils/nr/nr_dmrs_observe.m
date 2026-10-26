function obs = nr_dmrs_observe(H, env, F)
    %NR_DMRS_OBSERVE Received precoded DMRS of every scheduled UE, grouped by PRG
    %
    % Each layer's DMRS travels through the same precoder as its data, so the
    % observation reveals the channel only inside the span of the deployed
    % factors. The factor may be wideband or one per PRG
    %
    % Args:
    %   H: Slot response from nr_channel_slot
    %   env: Environment from nr_setup
    %   F: 1-by-U cell of transmit factors including power, each Lu-by-ru or
    %     Lu-by-ru-by-numGroups, with zero columns for an unscheduled UE
    %
    % Returns:
    %   obs: Per-PRG received samples Y, layer pilot symbols X, the per-PRG
    %     layer factor A with X observed through H * A, the layer layout, and
    %     each sample's subcarrier offset from the PRG centre in PRG widths

    U = env.U;
    positions = env.dmrs.positions;
    count = size(positions, 1);
    ranks = cellfun(@(f) size(f, 2), F);
    layerRows = cell(1, U);
    offset = 0;

    for u = 1:U
        layerRows{u} = offset + (1:ranks(u));
        offset = offset + ranks(u);
    end

    X = zeros(sum(ranks), count);

    for u = 1:U

        if ranks(u) > 0
            X(layerRows{u}, :) = transpose(env.dmrs.values{u, ranks(u)});
        end

    end

    obs = struct('ranks', ranks, 'layerRows', {layerRows});
    obs.Y = cell(1, env.numGroups);
    obs.X = cell(1, env.numGroups);
    obs.A = cell(1, env.numGroups);
    obs.offsets = cell(1, env.numGroups);
    linear = sub2ind([env.numSubcarriers, env.numSymbols], positions(:, 1), positions(:, 2));
    flat = reshape(H, size(H, 1), size(H, 2), []);

    for g = 1:env.numGroups
        A = zeros(env.Ltot, sum(ranks));

        for u = 1:U

            if ranks(u) > 0
                block = F{u}(:, :, min(g, size(F{u}, 3)));
                A(env.ranges{u}, layerRows{u}) = env.dmrsBoost * block;
            end

        end

        members = find(env.dmrs.groupOf == g);
        channel = flat(:, :, linear(members));
        S = A * X(:, members);
        Y = reshape(pagemtimes(channel, reshape(S, size(S, 1), 1, [])), size(H, 1), []);
        noise = (randn(size(Y)) + 1j * randn(size(Y))) / sqrt(2);
        obs.Y{g} = Y + env.noiseFactor * noise;
        obs.X{g} = X(:, members);
        obs.A{g} = A;
        obs.offsets{g} = frequency_offsets(positions(members, 1), env, g);
    end

end

function offsets = frequency_offsets(subcarriers, env, g)
    span = env.groups{g};
    offsets = (subcarriers(:)' - mean(span)) / numel(span);
end
