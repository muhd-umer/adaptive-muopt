function Q = ue_csirs_gram(obs, env, cfg)
    %UE_CSIRS_GRAM UE estimate of its per-PRG Grams from gNB-precoded CSI-RS
    %
    % The UE runs nrChannelEstimate with FD-CDM2 despreading inside each PRG,
    % since the precoding changes at PRG boundaries, and reads one estimate C
    % of the precoded channel per port, RB, and CSI-RS symbol. Its noise level
    % comes from the CSI-IM samples and the error variance of C from the
    % estimator's noise gain. Averaging conj(C * C') over a PRG estimates the
    % Gram plus numRx times that error variance, which is removed
    %
    % Args:
    %   obs: Observation from nr_csirs_observe
    %   env: Environment from nr_setup
    %   cfg: Configuration with csirs.averaging and prgRB
    %
    % Returns:
    %   Q: 1-by-K cell of Lu-by-Lu-by-G Gram estimates

    G = env.numGroups;
    noise = mean(abs(obs.im(:)) .^ 2);
    Q = cell(1, numel(obs.grids));

    for k = 1:numel(obs.grids)
        resource = env.csirs.resources{k};
        grid = obs.grids{k};
        Lu = size(grid, 3);
        estimate = nrChannelEstimate(env.carrier, grid, resource.linear, ...
            obs.amplitude * resource.values, 'CDMLengths', [2 1], ...
            'AveragingWindow', cfg.csirs.averaging, 'PRGBundleSize', cfg.prgRB, ...
            'Interpolation', 'off');
        ports = size(estimate, 4);
        C = zeros(Lu, ports, numel(resource.reads{1}));

        for p = 1:ports
            plane = reshape(estimate(:, :, :, p), [], Lu);

            for o = 1:size(C, 3)
                C(:, p, o) = mean(plane(resource.reads{p}{o}, :), 1).';
            end

        end

        bias = ports * noise * resource.noiseGain / obs.amplitude ^ 2 * eye(Lu);
        Q{k} = zeros(Lu, Lu, G);

        for g = 1:G
            samples = C(:, :, resource.readGroup == g);
            moment = conj(sum(pagemtimes(samples, 'none', samples, 'ctranspose'), 3)) / ...
                size(samples, 3) - bias;
            [W, E] = eig(0.5 * (moment + moment'));
            Q{k}(:, :, g) = W * diag(max(real(diag(E)), 0)) * W';
        end

    end

end
