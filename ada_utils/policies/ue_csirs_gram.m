function Q = ue_csirs_gram(obs, env, cfg, recordAfter)
    %UE_CSIRS_GRAM UE estimate of its per-PRG Grams from the gNB-precoded CSI-RS
    %
    % On each resource the UE runs nrChannelEstimate with CDM despreading
    % inside each PRG, since the precoding changes at PRG boundaries, and
    % reads one estimate C of the precoded channel per port, RB, and CSI-RS
    % symbol. Its noise level comes from the CSI-IM samples and the error
    % variance of C from the estimator's noise gain. Averaging conj(C * C')
    % over a PRG estimates the resource's Gram plus ports times that error
    % variance, which is removed. The cell-wide Gram less the update Grams
    % received so far is the Gram of each SIC term
    %
    % Args:
    %   obs: Observation from nr_csirs_observe
    %   env: Environment from nr_setup
    %   cfg: Configuration with csirs.averaging and prgRB
    %   recordAfter: Number of updates before each term the UE decodes within
    %
    % Returns:
    %   Q: 1-by-K cell of Lu-by-Lu-by-G Gram estimates, one per term

    noise = mean(abs(obs.im(:)) .^ 2);
    moments = cell(1, numel(obs.grids));

    for k = 1:numel(obs.grids)
        moments{k} = moment(obs.grids{k}, obs.resources{k}, obs.amplitude, noise, env, cfg);
    end

    running = cumsum(cat(4, moments{2:end}, zeros(size(moments{1}))), 4);
    Q = cell(1, numel(recordAfter));

    for i = 1:numel(recordAfter)
        Q{i} = moments{1};

        if recordAfter(i) > 0
            Q{i} = Q{i} - running(:, :, :, recordAfter(i));
        end

        Q{i} = psd(Q{i});
    end

end

function M = moment(grid, resource, amplitude, noise, env, cfg)
    Lu = size(grid, 3);
    estimate = nrChannelEstimate(env.carrier, grid, resource.linear, ...
        amplitude * resource.values, 'CDMLengths', resource.cdm, ...
        'AveragingWindow', cfg.csirs.averaging, 'PRGBundleSize', cfg.prgRB, ...
        'Interpolation', 'off');
    C = zeros(Lu, resource.ports, numel(resource.reads{1}));

    for p = 1:resource.ports
        plane = reshape(estimate(:, :, :, p), [], Lu);

        for o = 1:size(C, 3)
            C(:, p, o) = mean(plane(resource.reads{p}{o}, :), 1).';
        end

    end

    bias = resource.ports * noise * resource.noiseGain / amplitude ^ 2 * eye(Lu);
    M = zeros(Lu, Lu, env.numGroups);

    for g = 1:env.numGroups
        samples = C(:, :, resource.readGroup == g);
        M(:, :, g) = conj(sum(pagemtimes(samples, 'none', samples, 'ctranspose'), 3)) / ...
            size(samples, 3) - bias;
    end

end

function Q = psd(Q)

    for g = 1:size(Q, 3)
        [W, E] = eig(0.5 * (Q(:, :, g) + Q(:, :, g)'));
        Q(:, :, g) = W * diag(max(real(diag(E)), 0)) * W';
    end

end
