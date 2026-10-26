function obs = nr_srs_observe(H, env, active)
    %NR_SRS_OBSERVE Received non-precoded SRS of every scheduled UE, per comb offset and PRG
    %
    % Every UE sounds all of its ports at once with its own cyclic shifts, so
    % each SRS RE of a comb offset is one joint sample of the streams of the
    % UEs on that offset. The comb concentrates the sounding energy on 1/KTC
    % of the subcarriers
    %
    % Args:
    %   H: Slot response of the SRS slot from nr_channel_slot
    %   env: Environment from nr_setup
    %   active: Logical vector of UEs that sound in this slot
    %
    % Returns:
    %   obs: 1-by-S struct array, one per comb offset, with the per-PRG
    %     received samples Y, pilot symbols X over all Ltot streams, sample
    %     offsets from the PRG centre in PRG widths, the sounding factor A
    %     with X observed through H * A, and the rows silent on that offset

    obs = struct('Y', {}, 'X', {}, 'offsets', {}, 'A', {}, 'silentRows', {});
    flat = reshape(H, size(H, 1), size(H, 2), []);

    for group = env.srs.groups
        sounding = group.ues(active(group.ues));
        count = size(group.positions, 1);
        X = zeros(env.Ltot, count);
        A = zeros(env.Ltot);

        for u = group.ues
            X(env.ranges{u}, :) = transpose(env.srs.values{u});
        end

        for u = sounding
            A(env.ranges{u}, env.ranges{u}) = sqrt(env.srs.portEnergy(u)) * eye(env.Lxu(u));
        end

        linear = sub2ind([env.numSubcarriers, env.numSymbols], group.positions(:, 1), ...
            group.positions(:, 2));
        S = A * X;
        Y = reshape(pagemtimes(flat(:, :, linear), reshape(S, env.Ltot, 1, [])), size(H, 1), []);
        Y = Y + env.noiseFactor * (randn(size(Y)) + 1j * randn(size(Y))) / sqrt(2);
        entry = struct('Y', {cell(1, env.numGroups)}, 'X', {cell(1, env.numGroups)}, ...
            'offsets', {cell(1, env.numGroups)}, 'A', A, ...
            'silentRows', setdiff(1:env.Ltot, cell2mat(env.ranges(sounding))));

        for g = 1:env.numGroups
            members = group.groupOf == g;
            entry.Y{g} = Y(:, members);
            entry.X{g} = X(:, members);
            entry.offsets{g} = (group.positions(members, 1)' - mean(env.groups{g})) / ...
                numel(env.groups{g});
        end

        obs(end + 1) = entry; %#ok<AGROW>
    end

end
