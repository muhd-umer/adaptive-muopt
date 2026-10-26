function obs = nr_csirs_observe(H, env, cfg, u, precoders, calibration)
    %NR_CSIRS_OBSERVE Downlink grid a UE receives on gNB-precoded CSI-RS in TDD
    %
    % The downlink channel is the transpose of the uplink one, up to the
    % residual per-antenna mismatch of the transmit and receive chains. Each
    % precoder X goes out as one numRx-port CSI-RS resource through
    % conj(X^(1/2)) at a fixed configured power, so the UE's estimate C of the
    % precoded channel gives conj(C * C') = H_u' * X * H_u. The UE also
    % receives the samples of one zero-power CSI-IM resource
    %
    % Args:
    %   H: Slot response of the downlink slot from nr_channel_slot
    %   env: Environment from nr_setup
    %   cfg: Configuration with snrDb and csirs.snrOffsetDb
    %   u: UE index
    %   precoders: 1-by-K cell of numRx-by-numRx-by-G matrices X
    %   calibration: Residual downlink-over-uplink chain gains with gnb
    %     (numRx-by-1) and ue (1-by-U cell of Lu-by-1)
    %
    % Returns:
    %   obs: Per-resource received grids, CSI-IM samples, and the CSI-RS
    %     amplitude the UE is configured with

    G = env.numGroups;
    columns = env.ranges{u};
    Lu = numel(columns);
    numRx = size(H, 1);
    cells = env.numSubcarriers * env.numSymbols;
    % FD-CDM2 despreading leaves one estimate per port per pair of REs, so this
    % amplitude gives each estimate the configured downlink SNR per port
    amplitude = sqrt(10 ^ ((cfg.snrDb + cfg.csirs.snrOffsetDb) / 10) * numRx / 2);
    flat = reshape(H(:, columns, :, :), numRx, Lu, []);
    obs = struct('amplitude', amplitude, 'grids', {cell(1, numel(precoders))});

    for k = 1:numel(precoders)
        factors = zeros(numRx, numRx, G);

        for g = 1:G
            [V, D] = eig(0.5 * (precoders{k}(:, :, g) + precoders{k}(:, :, g)'));
            factors(:, :, g) = conj(V * diag(sqrt(max(real(diag(D)), 0))) * V');
        end

        resource = env.csirs.resources{k};
        downlink = calibration.ue{u}(:) .* pagetranspose(flat(:, :, resource.cells)) .* ...
            calibration.gnb(:).';
        transmit = pagemtimes(factors(:, :, resource.cellGroup), ...
            reshape(amplitude * resource.portSymbols.', numRx, 1, []));
        received = reshape(pagemtimes(downlink, transmit), Lu, []);
        grid = (randn(cells, Lu) + 1j * randn(cells, Lu)) / sqrt(2);
        grid(resource.cells, :) = grid(resource.cells, :) + received.';
        obs.grids{k} = reshape(grid, env.numSubcarriers, env.numSymbols, Lu);
    end

    obs.im = (randn(Lu, env.csirs.imRE) + 1j * randn(Lu, env.csirs.imRE)) / sqrt(2);
end
