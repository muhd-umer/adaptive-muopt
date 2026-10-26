function obs = nr_dmrs_observe(H, env, F)
    %NR_DMRS_OBSERVE Received precoded DMRS of every scheduled UE, per CDM group and PRG
    %
    % Each layer's DMRS travels through the same precoder as its data, so the
    % observation reveals the channel only inside the span of the deployed
    % factors. The layers of the UEs in one CDM group share its REs and are
    % observed jointly; the other groups are silent there. The factor may be
    % wideband or one per PRG
    %
    % Args:
    %   H: Slot response from nr_channel_slot
    %   env: Environment from nr_setup
    %   F: 1-by-U cell of transmit factors including power, each Lu-by-ru or
    %     Lu-by-ru-by-numGroups, with zero columns for an unscheduled UE
    %
    % Returns:
    %   obs: Per-UE ranks and a 1-by-C struct array over CDM groups with the
    %     UEs, each UE's rows in the group's layers, and the per-PRG received
    %     samples Y, layer pilots X, layer factor A (X is observed through
    %     H * A), and sample offsets from the PRG centre in PRG widths

    ranks = cellfun(@(f) size(f, 2), F);
    obs = struct('ranks', ranks, 'cdm', struct('ues', {}, 'layerRows', {}, 'Y', {}, ...
        'X', {}, 'A', {}, 'offsets', {}));
    flat = reshape(H, size(H, 1), size(H, 2), []);

    for group = env.dmrs.cdm
        members = group.ues(ranks(group.ues) > 0);
        layerRows = cell(1, env.U);
        offset = 0;

        for u = members
            layerRows{u} = offset + (1:ranks(u));
            offset = offset + ranks(u);
        end

        X = zeros(offset, size(group.positions, 1));

        for u = members
            X(layerRows{u}, :) = transpose(env.dmrs.values{u, ranks(u)});
        end

        linear = sub2ind([env.numSubcarriers, env.numSymbols], group.positions(:, 1), ...
            group.positions(:, 2));
        entry = struct('ues', members, 'layerRows', {layerRows}, ...
            'Y', {cell(1, env.numGroups)}, 'X', {cell(1, env.numGroups)}, ...
            'A', {cell(1, env.numGroups)}, 'offsets', {cell(1, env.numGroups)});

        for g = 1:env.numGroups
            A = zeros(env.Ltot, offset);

            for u = members
                A(env.ranges{u}, layerRows{u}) = env.dmrs.boost * F{u}(:, :, min(g, size(F{u}, 3)));
            end

            inside = find(group.groupOf == g);
            S = A * X(:, inside);
            Y = reshape(pagemtimes(flat(:, :, linear(inside)), reshape(S, env.Ltot, 1, [])), ...
                size(H, 1), []);
            entry.Y{g} = Y + env.noiseFactor * (randn(size(Y)) + 1j * randn(size(Y))) / sqrt(2);
            entry.X{g} = X(:, inside);
            entry.A{g} = A;
            entry.offsets{g} = (group.positions(inside, 1)' - mean(env.groups{g})) / ...
                numel(env.groups{g});
        end

        obs.cdm(end + 1) = entry;
    end

end
