function [state, sends] = nr_codebook_policy(state, ctx, srs, dmrs, belief)
    %NR_CODEBOOK_POLICY gNB scheduler of codebook-based PUSCH precoders and TPC commands
    %
    % The 'csi' belief is the conventional baseline: at each SRS occasion a
    % per-PRG least-squares channel estimate with its noise bias removed
    % replaces the Gram belief, and the selector runs only then. The 'pilot'
    % belief never forms a channel matrix: full-port SRS yields MMSE
    % information that pins the whole Gram-equivalent belief, the precoded
    % DMRS of every slot refreshes it inside the excited subspace with
    % exponential forgetting, and an update is sent only when its predicted
    % gain beats the trigger. Observations of one comb offset or CDM group are
    % joint, so cross terms between UEs come only from a shared one. The MMSE
    % receiver has one tap per gNB antenna, so when a PRG holds too few
    % samples for it, adjacent PRGs are learned together
    %
    % Args:
    %   state: Codebook method state with the belief K and the planned actions
    %   ctx: Run context with env, selection spec, and policy settings
    %   srs: SRS observations since the last uplink slot, one per comb offset,
    %     or empty
    %   dmrs: DMRS observation of this slot through the deployed actions
    %   belief: 'csi' or 'pilot'
    %
    % Returns:
    %   state: Updated belief and evaluation count
    %   sends: Struct array of UE, catalog row, TPC code point, and power

    env = ctx.env;
    policy = ctx.policy;
    sends = struct('ue', {}, 'row', {}, 'tpc', {}, 'powerDb', {});

    switch belief
        case 'csi'

            if isempty(srs)
                return;
            end

            for g = 1:env.numGroups
                state.K(:, :, g) = estimated_gram(arrayfun(@(o) o.Y{g}, srs, 'UniformOutput', ...
                    false), arrayfun(@(o) o.X{g}, srs, 'UniformOutput', false), {srs.A});
            end

        case 'pilot'

            for o = reshape(srs, 1, [])
                A = repmat({o.A}, 1, env.numGroups);
                state.K = fuse(state.K, o, A, o.silentRows, policy.srsForget);
                state.ready = true;
            end

            if ~state.ready
                return;
            end

            for o = dmrs.cdm
                state.K = fuse(state.K, o, o.A, [], policy.dmrsForget);
            end

    end

    state.ready = true;
    opts = ctx.selectOpts;
    opts.reachable = 'tpc';
    [actions, out] = nr_select(state.K, env.catalogs, state.planned, ctx.selection, opts);
    state.evaluations = state.evaluations + out.evaluations;

    if strcmp(belief, 'pilot') && ...
            out.gain <= policy.triggerRelative * max(abs(out.currentValue), 1e-9)
        return;
    end

    for u = find([actions.changed])
        sends(end + 1) = struct('ue', u, 'row', actions(u).row, 'tpc', actions(u).tpc, ...
            'powerDb', actions(u).powerDb); %#ok<AGROW>
    end

end

function K = fuse(K, obs, A, silentRows, beta)
    %FUSE Fold the MMSE information of one joint observation into every PRG's belief
    G = numel(obs.Y);
    [numRx, count] = size(obs.Y{1});
    slope = count >= 6 * numRx;
    bundle = max(1, ceil(3 * numRx / count));

    for first = 1:bundle:G
        bands = first:min(first + bundle - 1, G);

        if slope
            J = mmse_info(obs.Y{first}, obs.X{first}, struct('silentRows', silentRows, ...
                'offsets', obs.offsets{first}));
        else
            J = mmse_info([obs.Y{bands}], [obs.X{bands}], struct('silentRows', silentRows));
        end

        for g = bands
            K(:, :, g) = gram_fuse(K(:, :, g), J, A{g}, beta);
        end

    end

end

function K = estimated_gram(Y, X, A)
    %ESTIMATED_GRAM Bias-corrected Gram matrix from least-squares estimates of each comb offset
    %
    % With white noise of power s2 the LS estimate C of the streams sounded on
    % one offset satisfies E[C' * C] = H' * H + Ly * s2 * inv(S * S'), so that
    % term is removed from its diagonal block; cross blocks between offsets
    % carry no noise bias. The result is normalized by the pooled noise power
    %
    % Args:
    %   Y: 1-by-S cell of received samples, one per comb offset
    %   X: 1-by-S cell of pilot symbols over all streams
    %   A: 1-by-S cell of sounding factors
    %
    % Returns:
    %   K: Ltot-by-Ltot Gram estimate
    Ltot = size(A{1}, 2);
    numRx = size(Y{1}, 1);
    channel = zeros(numRx, Ltot);
    bias = zeros(Ltot);
    residualEnergy = 0;
    dof = 0;

    for s = 1:numel(Y)
        excited = find(vecnorm(A{s}, 2, 1) > 0);

        if isempty(excited)
            continue;
        end

        S = A{s}(excited, excited) * X{s}(excited, :);
        gramS = S * S';
        estimate = (Y{s} * S') / gramS;
        residual = Y{s} - estimate * S;
        channel(:, excited) = estimate;
        bias(excited, excited) = numRx * inv(gramS);
        residualEnergy = residualEnergy + real(residual(:)' * residual(:));
        dof = dof + numRx * max(size(X{s}, 2) - numel(excited), 1);
    end

    if dof == 0
        K = zeros(Ltot);
        return;
    end

    noise = max(residualEnergy / dof, 1e-12);
    unbiased = (channel' * channel) / noise - bias;
    [V, D] = eig(0.5 * (unbiased + unbiased'));
    K = V * diag(max(real(diag(D)), 0)) * V';
    K = 0.5 * (K + K');
end
