%ADA_NRRECIPROCITY How the reciprocity methods degrade with CSI-RS timing, calibration, and mobility
%
% The default experiment sends TS 38.211 CSI-RS in the second downlink slot
% after the uplink DMRS slot of the DDDSU pattern, the latest that leaves
% the UE 42 symbols before the next uplink slot, estimated by the UE with
% nrChannelEstimate and a CSI-IM noise measurement, over chains with a
% 0.5 dB, 5 degree rms residual calibration error. Each case changes one of
% those settings: more CSI-RS symbols, a later CSI-RS slot, the calibration
% error, and mobility. Every case reports the paired maxRMAC gap to the canonical
% optimum

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'ada_utils')), fullfile(root, 'maxresmac'), ...
    fullfile(root, 'maxrmac'));

base = nr_default_config();
base.realizations = 40;
base.methods = {'canonical', 'nr-oracle', 'ue-pilot', 'ue-pilot-free'};

if strcmp(getenv('ADA_QUICK'), '1')
    base.realizations = 3;
    base.horizonSlots = 200;
    base.warmupSlots = 50;
end

calibrated = @(db, deg) {'csirs.calibration', struct('amplitudeDb', db, 'phaseDeg', deg)};
cases = struct('label', {}, 'changes', {});
cases(end + 1) = struct('label', 'default', 'changes', {{}});
cases(end + 1) = struct('label', 'CSI-RS 1 slot after DMRS', 'changes', ...
    {{'csirs.lagSlots', 1}});
cases(end + 1) = struct('label', 'UE processing 2 slots, CSI-RS 3 after', 'changes', ...
    {{'csirs.lagSlots', 3, 'csirs.processingSlots', 2}});
cases(end + 1) = struct('label', 'CSI-RS in S slot', 'changes', {{'csirs.lagSlots', 4}});
cases(end + 1) = struct('label', 'perfect calibration', 'changes', {calibrated(0, 0)});
cases(end + 1) = struct('label', 'calibration 1 dB, 10 deg', 'changes', {calibrated(1, 10)});
cases(end + 1) = struct('label', 'calibration 2 dB, 20 deg', 'changes', {calibrated(2, 20)});
cases(end + 1) = struct('label', '3 km/h', 'changes', {{'channel.speedKmh', 3}});
cases(end + 1) = struct('label', '60 km/h', 'changes', {{'channel.speedKmh', 60}});

spec = struct('type', 'maxrmac', 'theta', base.theta);
fprintf('[maxrmac reciprocity realism, weighted-rate gap to canonical]\n');
fprintf('%-30s', 'case');

for m = 2:numel(base.methods)
    fprintf(' %18s', base.methods{m});
end

fprintf('\n%s\n', repmat('-', 1, 30 + 19 * (numel(base.methods) - 1)));
table = struct('label', {}, 'gap', {}, 'ci', {});

for c = 1:numel(cases)
    cfg = base;

    for k = 1:2:numel(cases(c).changes)
        parts = strsplit(cases(c).changes{k}, '.');
        cfg = setfield(cfg, parts{:}, cases(c).changes{k + 1});
    end

    evalc('[results, env] = nr_experiment(cfg, spec);');
    evalc('stats = nr_summary(results, cfg, env, {''canonical''}, cases(c).label);');
    gap = stats.gaps.canonical;
    table(end + 1) = struct('label', cases(c).label, 'gap', gap.mean, 'ci', gap.ci); %#ok<SAGROW>
    fprintf('%-30s', cases(c).label);

    for m = 2:numel(base.methods)
        fprintf(' %+8.2f%%+-%5.2f%%', 100 * gap.mean(m), 100 * gap.ci(m));
    end

    fprintf('\n');
end

outputDir = fullfile(root, 'experiments', 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'nrReciprocity.mat'), 'base', 'cases', 'table');
