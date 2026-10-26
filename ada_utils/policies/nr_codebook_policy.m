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
    % gain beats the trigger
    %
    % Args:
    %   state: Codebook method state with the belief K and the planned actions
    %   ctx: Run context with env, selection spec, and policy settings
    %   srs: SRS observation since the last uplink slot, or empty
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
                state.K(:, :, g) = estimated_gram(srs.Y{g}, srs.X{g}, srs.A);
            end

        case 'pilot'

            if ~isempty(srs)

                for g = 1:env.numGroups
                    J = mmse_info(srs.Y{g}, srs.X{g}, struct('silentRows', srs.silentRows, ...
                        'offsets', srs.offsets{g}));
                    state.K(:, :, g) = gram_fuse(state.K(:, :, g), J, srs.A, policy.srsForget);
                end

                state.ready = true;
            end

            if ~state.ready
                return;
            end

            if sum(dmrs.ranks) > 0

                for g = 1:env.numGroups
                    J = mmse_info(dmrs.Y{g}, dmrs.X{g}, struct('offsets', dmrs.offsets{g}));
                    state.K(:, :, g) = gram_fuse(state.K(:, :, g), J, dmrs.A{g}, ...
                        policy.dmrsForget);
                end

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

function K = estimated_gram(Y, X, A)
    %ESTIMATED_GRAM Bias-corrected Gram matrix of a least-squares channel estimate
    %
    % With white noise of power s2 the LS estimate C satisfies
    % E[C' * C] = H' * H + Ly * s2 * inv(S * S'), so that term is removed
    % before normalizing by the estimated noise power
    excited = vecnorm(A, 2, 1) > 0;
    K = zeros(size(A, 2));

    if ~any(excited)
        return;
    end

    S = A(excited, excited) * X(excited, :);
    gramS = S * S';
    estimate = (Y * S') / gramS;
    residual = Y - estimate * S;
    noise = max(real(residual(:)' * residual(:)) / ...
        (size(Y, 1) * max(size(X, 2) - nnz(excited), 1)), 1e-12);
    unbiased = (estimate' * estimate - size(Y, 1) * noise * inv(gramS)) / noise;
    [V, D] = eig(0.5 * (unbiased + unbiased'));
    K(excited, excited) = V * diag(max(real(diag(D)), 0)) * V';
    K = 0.5 * (K + K');
end
