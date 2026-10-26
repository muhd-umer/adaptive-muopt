function obs = nr_srs_observe(H, env, active)
    %NR_SRS_OBSERVE Received non-precoded SRS of every scheduled UE, grouped by PRG
    %
    % Every UE sounds all of its ports at once with its own cyclic shift on
    % the shared comb, so each SRS RE is one joint sample of all streams. The
    % comb concentrates the sounding energy on 1/KTC of the subcarriers
    %
    % Args:
    %   H: Slot response from nr_channel_slot
    %   env: Environment from nr_setup
    %   active: Logical vector of UEs that sound in this slot
    %
    % Returns:
    %   obs: Per-PRG received samples Y and pilot symbols X, the sounding factor
    %     A with X observed through H * A, the silent stream rows, and each
    %     sample's subcarrier offset from the PRG centre in PRG widths

    U = env.U;
    positions = env.srs.positions;
    count = size(positions, 1);
    X = zeros(env.Ltot, count);
    A = zeros(env.Ltot);
    silent = [];

    for u = 1:U
        rows = env.ranges{u};
        X(rows, :) = transpose(env.srs.values{u});

        if active(u)
            A(rows, rows) = sqrt(env.srs.portEnergy(u)) * eye(numel(rows));
        else
            silent = [silent, rows]; %#ok<AGROW>
        end

    end

    Y = received(H, positions, A * X, env);
    obs = struct('A', A, 'silentRows', silent);
    obs.Y = cell(1, env.numGroups);
    obs.X = cell(1, env.numGroups);
    obs.offsets = cell(1, env.numGroups);

    for g = 1:env.numGroups
        members = env.srs.groupOf == g;
        obs.Y{g} = Y(:, members);
        obs.X{g} = X(:, members);
        obs.offsets{g} = (positions(members, 1)' - mean(env.groups{g})) / numel(env.groups{g});
    end

end

function Y = received(H, positions, S, env)
    count = size(positions, 1);
    linear = sub2ind([env.numSubcarriers, env.numSymbols], positions(:, 1), positions(:, 2));
    flat = reshape(H, size(H, 1), size(H, 2), []);
    channel = flat(:, :, linear);
    Y = squeeze(pagemtimes(channel, reshape(S, size(S, 1), 1, count)));

    if size(H, 1) == 1
        Y = reshape(Y, 1, count);
    end

    noise = (randn(size(H, 1), count) + 1j * randn(size(H, 1), count)) / sqrt(2);
    Y = Y + env.noiseFactor * noise;
end
