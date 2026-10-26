function obs = nr_csirs_observe(H, env, cfg, u, plan, calibration)
    %NR_CSIRS_OBSERVE Downlink grids a UE receives on the gNB-precoded CSI-RS in TDD
    %
    % The downlink channel is the transpose of the uplink one, up to the
    % residual per-antenna mismatch of the transmit and receive chains. The
    % cell-wide resource carries conj(base^(1/2)) and resource pair{u, j}
    % carries conj(update), so the UE's estimate C of each precoded channel
    % gives conj(C * C') = H_u' * P * P' * H_u for that resource's precoder P.
    % The UE also receives the samples of the zero-power CSI-IM resource
    %
    % Args:
    %   H: Slot response of the downlink slot from nr_channel_slot
    %   env: Environment from nr_setup
    %   cfg: Configuration with snrDb and csirs.snrOffsetDb
    %   u: UE index
    %   plan: Reciprocity plan from nr_recip_policy
    %   calibration: Residual downlink-over-uplink chain gains with gnb
    %     (numRx-by-1) and ue (1-by-U cell of Lu-by-1)
    %
    % Returns:
    %   obs: Received grids of the cell-wide resource and of each update in
    %     order, their resources, CSI-IM samples, and the per-port CSI-RS
    %     amplitude the UE is configured with

    columns = env.ranges{u};
    Lu = numel(columns);
    numRx = env.numRx;
    cells = env.numSubcarriers * env.numSymbols;
    amplitude = sqrt(10 ^ ((cfg.snrDb + cfg.csirs.snrOffsetDb) / 10));
    flat = reshape(H(:, columns, :, :), numRx, Lu, []);
    precoders = [{root(plan.base)}, plan.updates{u}];
    resources = [{env.csirs.base}, env.csirs.pair(u, plan.others{u}(1:numel(plan.updates{u})))];
    obs = struct('amplitude', amplitude, 'resources', {resources}, ...
        'grids', {cell(1, numel(precoders))});

    for k = 1:numel(precoders)
        resource = resources{k};
        downlink = calibration.ue{u}(:) .* pagetranspose(flat(:, :, resource.cells)) .* ...
            calibration.gnb(:).';
        transmit = pagemtimes(conj(precoders{k}(:, :, resource.cellGroup)), ...
            reshape(amplitude * resource.portSymbols.', resource.ports, 1, []));
        received = reshape(pagemtimes(downlink, transmit), Lu, []);
        grid = (randn(cells, Lu) + 1j * randn(cells, Lu)) / sqrt(2);
        grid(resource.cells, :) = grid(resource.cells, :) + received.';
        obs.grids{k} = reshape(grid, env.numSubcarriers, env.numSymbols, Lu);
    end

    obs.im = (randn(Lu, env.csirs.imRE) + 1j * randn(Lu, env.csirs.imRE)) / sqrt(2);
end

function factor = root(X)
    factor = zeros(size(X));

    for g = 1:size(X, 3)
        [V, D] = eig(0.5 * (X(:, :, g) + X(:, :, g)'));
        factor(:, :, g) = V * diag(sqrt(max(real(diag(D)), 0))) * V';
    end

end
