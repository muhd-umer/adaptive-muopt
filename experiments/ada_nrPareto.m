%ADA_NRPARETO Performance versus control signaling for the NR-native and reciprocity modes
%
% Sweeps the knobs that trade signaling for performance on maxRMAC and
% maxRESMAC: the SRS period and the objective-aware trigger of the NR-native
% pilot policy, the SRS period of the estimated-CSI baseline, and the
% CSI-RS power of the reciprocity-based UE-side solver. SRS periods are
% whole TDD periods so that every sounding falls in an S slot. Every point
% reports its
% gap to the perfect-CSI NR oracle of the same channels, its design-field
% and full-DCI bit rates, its downlink CSI-RS load, and its SRS-inclusive net
% throughput

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'ada_utils')), fullfile(root, 'maxresmac'), ...
    fullfile(root, 'maxrmac'));

base = nr_default_config();
base.realizations = 12;
base.horizonSlots = 300;
base.warmupSlots = 75;

if strcmp(getenv('ADA_QUICK'), '1')
    base.realizations = 3;
    base.horizonSlots = 200;
    base.warmupSlots = 50;
end

objectives = {struct('type', 'maxrmac', 'theta', base.theta), ...
    struct('type', 'maxresmac', 'theta', base.theta, 'sumFraction', 0.75)};
points = struct('label', {}, 'method', {}, 'changes', {});
points(end + 1) = struct('label', 'nr-csi srs=5', 'method', 'nr-csi', 'changes', {{'srs.periodSlots', 5}});
points(end + 1) = struct('label', 'nr-csi srs=10', 'method', 'nr-csi', 'changes', {{'srs.periodSlots', 10}});
points(end + 1) = struct('label', 'nr-csi srs=20', 'method', 'nr-csi', 'changes', {{'srs.periodSlots', 20}});

for period = [5 10 20]

    for trigger = [0 0.01 0.05]
        points(end + 1) = struct('label', sprintf('nr-pilot srs=%d trig=%.2f', period, trigger), ...
            'method', 'nr-pilot', 'changes', ...
            {{'srs.periodSlots', period, 'policy.triggerRelative', trigger}}); %#ok<SAGROW>
    end

end

points(end + 1) = struct('label', 'ue-pilot csirs SNR -10 dB', 'method', 'ue-pilot', ...
    'changes', {{'csirs.snrOffsetDb', 0}});

for o = 1:numel(objectives)
    spec = objectives{o};
    fprintf('\n[%s performance versus signaling]\n', spec.type);
    fprintf('%-30s %12s %16s %12s %12s %12s %12s\n', 'configuration', 'objective', ...
        'gap to oracle', 'design b/s', 'DCI b/s', 'CSI-RS RE/s', 'net thr');
    fprintf('%s\n', repmat('-', 1, 113));
    frontier = struct('label', {}, 'objective', {}, 'gap', {}, 'gapCI', {}, ...
        'designRate', {}, 'controlRate', {}, 'csirsRate', {}, 'netThroughput', {});

    for p = 1:numel(points)
        cfg = base;
        cfg.methods = {'nr-oracle', points(p).method};

        for k = 1:2:numel(points(p).changes)
            cfg = set_field(cfg, points(p).changes{k}, points(p).changes{k + 1});
        end

        evalc('[results, env] = nr_experiment(cfg, spec);');
        evalc('stats = nr_summary(results, cfg, env, {''nr-oracle''}, points(p).label);');
        gap = stats.gaps.nr_oracle;
        frontier(end + 1) = struct('label', points(p).label, 'objective', stats.objective(2), ...
            'gap', gap.mean(2), 'gapCI', gap.ci(2), 'designRate', stats.designRate(2), ...
            'controlRate', stats.controlRate(2), 'csirsRate', stats.csirsRate(2), ...
            'netThroughput', stats.netThroughput(2)); %#ok<SAGROW>
        fprintf('%-30s %12.4f %+8.2f%%+-%4.2f%% %12.0f %12.0f %12.0f %12.4f\n', points(p).label, ...
            stats.objective(2), 100 * gap.mean(2), 100 * gap.ci(2), stats.designRate(2), ...
            stats.controlRate(2), stats.csirsRate(2), stats.netThroughput(2));
    end

    pareto.(spec.type) = frontier;
end

fprintf('\n  gaps are paired against the perfect-CSI NR oracle of the same channels;\n');
fprintf('  ue-pilot needs TDD reciprocity.\n');

outputDir = fullfile(root, 'experiments', 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'nrPareto.mat'), 'base', 'objectives', 'points', 'pareto');

function cfg = set_field(cfg, path, value)
    parts = strsplit(path, '.');
    cfg = setfield(cfg, parts{:}, value);
end
