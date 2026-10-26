function stats = nr_summary(results, cfg, env, references, label)
    %NR_SUMMARY Paired statistics across realizations and the printed experiment report
    %
    % Every metric is averaged over the uplink slots after the warm-up within
    % each realization, then summarized across realizations with a 95%
    % confidence interval; rates per second count the whole post-warm-up
    % time, downlink slots included. Gaps to a reference are formed per
    % realization before averaging, so they are paired over identical
    % channels and noise
    %
    % Args:
    %   results: Cell array of nr_run outputs, one per realization
    %   cfg: Configuration used for the runs
    %   env: Environment used for the runs
    %   references: Cell array of method names to report gaps against
    %   label: Experiment title
    %
    % Returns:
    %   stats: Per-method means and confidence half-widths of every metric,
    %     the per-reference paired gaps, per-slot traces, and the count of
    %     realizations in which a method beats the canonical optimum

    names = results{1}.names;
    spec = results{1}.spec;
    M = numel(names);
    R = numel(results);
    window = find(results{1}.slots >= cfg.warmupSlots);
    duration = (cfg.horizonSlots - cfg.warmupSlots) * env.slotDuration;
    U = env.U;
    theta = spec.theta(:);
    stats = struct('names', {names}, 'type', spec.type, 'realizations', R);
    fields = {'objective', 'throughput', 'netThroughput', 'designRate', 'controlRate', ...
        'messageRate', 'csirsRate', 'energy', 'runtime', 'lost', 'nackRate'};
    perRealization = struct();

    for f = 1:numel(fields)
        perRealization.(fields{f}) = nan(M, R);
    end

    userRates = nan(M, U, R);
    userEnergy = nan(M, U, R);

    for r = 1:R

        for m = 1:M
            metrics = results{r}.metrics{m};
            energy = metrics.energy(:, window);
            thr = metrics.throughput(:, window);
            fraction = metrics.dataFraction(window);
            userRates(m, :, r) = mean(metrics.rates(:, window), 2);
            userEnergy(m, :, r) = mean(energy, 2);
            perRealization.objective(m, r) = mean(metrics.objective(window));
            perRealization.energy(m, r) = sum(mean(energy, 2));
            perRealization.throughput(m, r) = theta' * mean(thr, 2, 'omitnan');
            perRealization.netThroughput(m, r) = theta' * mean(thr .* fraction, 2, 'omitnan');
            perRealization.designRate(m, r) = sum(metrics.designBits(window)) / duration;
            perRealization.controlRate(m, r) = sum(metrics.dciBits(window)) / duration;
            perRealization.messageRate(m, r) = sum(metrics.messages(window)) / duration;
            perRealization.csirsRate(m, r) = sum(metrics.csirsRE(window)) / duration;
            perRealization.runtime(m, r) = mean(metrics.runtimeMs(window));
            perRealization.lost(m, r) = sum(metrics.lost(window));
            planned = metrics.planned(:, window);
            perRealization.nackRate(m, r) = sum(metrics.nack(window)) / ...
                max(nnz(planned > 0), 1);
        end

    end

    for f = 1:numel(fields)
        values = perRealization.(fields{f});
        stats.(fields{f}) = mean(values, 2, 'omitnan');
        stats.([fields{f}, 'CI']) = 1.96 * std(values, 0, 2, 'omitnan') ./ ...
            sqrt(max(sum(~isnan(values), 2), 1));
    end

    stats.csirsShare = stats.csirsRate / env.tdd.downlinkRE;
    stats.userRates = mean(userRates, 3, 'omitnan');
    stats.userEnergy = mean(userEnergy, 3, 'omitnan');
    stats.trace = slot_traces(results, theta);
    stats.perRealization = perRealization;
    stats.gaps = struct();

    for k = 1:numel(references)
        reference = find(strcmp(names, references{k}), 1);

        if isempty(reference)
            continue;
        end

        base = perRealization.objective(reference, :);
        gap = (base - perRealization.objective) ./ abs(base);
        key = matlab.lang.makeValidName(references{k});
        stats.gaps.(key) = struct('mean', mean(gap, 2, 'omitnan'), ...
            'ci', 1.96 * std(gap, 0, 2, 'omitnan') / sqrt(R));
    end

    stats.boundViolations = bound_violations(perRealization.objective, names);
    print_report(stats, names, references, label, cfg, env, spec);
end

function violations = bound_violations(objective, names)
    %BOUND_VIOLATIONS Realizations in which a method beats the canonical optimum
    %
    % The canonical design is the per-slot optimum under the same budgets on
    % the same subcarriers, so a method that beats it points to a modeling bug
    violations = zeros(numel(names), 1);
    reference = find(strcmp(names, 'canonical'), 1);

    if isempty(reference)
        return;
    end

    violations = sum(objective > objective(reference, :) * (1 + 2e-3), 2);
    violations(reference) = 0;

    for m = find(violations(:)')
        warning('nr_summary:boundViolation', ...
            '%s beats the canonical optimum in %d realizations; check the model', ...
            names{m}, violations(m));
    end

end

function trace = slot_traces(results, theta)
    %SLOT_TRACES Per-slot means and 95% half-widths across realizations
    %
    % Returns the objective, the weighted link throughput, the design bits,
    % and the control messages of every method in every slot, so the time
    % evolution of rate and feedback can be plotted after the run
    R = numel(results);
    M = numel(results{1}.metrics);
    S = numel(results{1}.metrics{1}.objective);
    names = {'objective', 'throughput', 'designBits', 'messages'};
    trace = struct('slots', results{1}.slots);

    for f = 1:numel(names)
        values = nan(M, S, R);

        for r = 1:R

            for m = 1:M
                metrics = results{r}.metrics{m};

                if strcmp(names{f}, 'throughput')
                    values(m, :, r) = theta' * metrics.throughput;
                else
                    values(m, :, r) = metrics.(names{f});
                end

            end

        end

        trace.(names{f}) = mean(values, 3, 'omitnan');
        trace.([names{f}, 'CI']) = 1.96 * std(values, 0, 3, 'omitnan') ./ ...
            sqrt(max(sum(~isnan(values), 3), 1));
    end

end

function print_report(stats, names, references, label, cfg, env, spec)
    fprintf('\n[%s]\n', label);
    fprintf(['  %d realizations, %d slots each (%d warm-up), TDD %s, %d kHz SCS, %d RB, ' ...
        '%d PRGs\n'], stats.realizations, cfg.horizonSlots, cfg.warmupSlots, env.tdd.pattern, ...
        cfg.carrier.scs, cfg.carrier.NRB, env.numGroups);
    fprintf(['  gNB panel [%s] (%d antennas), UE ports [%s], SNR %g dB, gains [%s] dB, ' ...
        '%s %g ns, %s km/h\n'], num2str(cfg.gnbArray), env.numRx, num2str(env.Lxu), cfg.snrDb, ...
        num2str(cfg.userGainDb), cfg.channel.delayProfile, 1e9 * cfg.channel.delaySpread, ...
        num2str(cfg.channel.speedKmh));
    fprintf('  SRS every %d slots, control delay %d slots, DCI precoding fields [%s] bits\n', ...
        cfg.srs.periodSlots, cfg.control.delay, num2str(env.dci.precoding));
    fprintf(['  CSI-RS at least %d slot(s) after the DMRS, UE processing %d slots, ' ...
        'calibration %g dB / %g deg rms\n'], cfg.csirs.lagSlots, cfg.csirs.processingSlots, ...
        cfg.csirs.calibration.amplitudeDb, cfg.csirs.calibration.phaseDeg);
    fprintf('  weights [%s]\n\n', num2str(spec.theta(:)', '%.2f '));
    fprintf('%-13s %15s %15s %15s %10s %10s %11s %7s %7s %8s\n', 'method', 'w-rate', ...
        'w-throughput', 'net w-thr', 'design b/s', 'DCI b/s', 'CSI-RS RE/s', 'DL %', ...
        'NACK', 'ms/slot');
    fprintf('%s\n', repmat('-', 1, 124));

    for m = 1:numel(names)
        fprintf(['%-13s %8.3f+-%5.3f %8.3f+-%5.3f %8.3f+-%5.3f %10.0f %10.0f %11.0f ' ...
            '%7.2f %7.3f %8.1f\n'], names{m}, stats.objective(m), stats.objectiveCI(m), ...
            stats.throughput(m), stats.throughputCI(m), stats.netThroughput(m), ...
            stats.netThroughputCI(m), stats.designRate(m), stats.controlRate(m), ...
            stats.csirsRate(m), 100 * stats.csirsShare(m), stats.nackRate(m), ...
            stats.runtime(m));
    end

    if any(stats.boundViolations)
        fprintf('\n  WARNING: some methods beat the canonical optimum; results are not valid\n');
    elseif any(strcmp(names, 'canonical'))
        fprintf('\n  sanity: no method beats the canonical optimum in any realization\n');
    end

    keys = fieldnames(stats.gaps);

    for k = 1:numel(keys)
        gap = stats.gaps.(keys{k});
        fprintf('\n  weighted-rate gap to %s (paired, 95%% CI)\n', references{k});

        for m = 1:numel(names)
            fprintf('    %-13s %+7.2f%% +- %5.2f%%\n', names{m}, 100 * gap.mean(m), ...
                100 * gap.ci(m));
        end

    end

    fprintf('\n  design b/s counts the transmit-action fields (precoding, TPC, SRI) or the\n');
    fprintf('  custom covariance payload; DCI b/s counts every carrying DCI 0_1 including\n');
    fprintf('  its other fields and CRC. CSI-RS RE/s is the downlink sounding the\n');
    fprintf('  reciprocity-based ue-pilot spends instead of precoder feedback, and DL %%\n');
    fprintf('  is its share of the downlink resource elements of the TDD pattern. Genie\n');
    fprintf('  references (canonical, nr-oracle, hold) signal nothing and plan their MCS\n');
    fprintf('  from the true rate.\n');
end
