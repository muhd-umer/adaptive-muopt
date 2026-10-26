%ADA_NRABLATION Robustness of the NR maxRMAC methods to SNR, mobility, and colored interference
%
% Each case changes one physical assumption of the default experiment and
% reports the paired gap of every method to the perfect-CSI NR oracle on the
% same channels. The colored-interference case adds a strong spatially
% correlated interferer to the receiver noise: the estimated-CSI baseline
% keeps its white-noise model, while the MMSE information whitens the
% observation without being told the noise covariance

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'ada_utils')), fullfile(root, 'maxresmac'), ...
    fullfile(root, 'maxrmac'));

base = nr_default_config();
base.realizations = 20;
base.horizonSlots = 300;
base.warmupSlots = 75;
base.methods = {'nr-oracle', 'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', 'hold'};

if strcmp(getenv('ADA_QUICK'), '1')
    base.realizations = 3;
    base.horizonSlots = 200;
    base.warmupSlots = 50;
end

rng(77);
steering = exp(1j * pi * (0:base.numRx - 1)' * sin(0.4));
interference = eye(base.numRx) + 10 * (steering * steering');
cases = struct('label', {}, 'changes', {});
cases(end + 1) = struct('label', 'default', 'changes', {{}});
cases(end + 1) = struct('label', 'snr 0 dB', 'changes', {{'snrDb', 0}});
cases(end + 1) = struct('label', 'snr 10 dB', 'changes', {{'snrDb', 10}});
cases(end + 1) = struct('label', '3 km/h', 'changes', {{'channel.speedKmh', [3 3]}});
cases(end + 1) = struct('label', '60 km/h', 'changes', {{'channel.speedKmh', [60 60]}});
cases(end + 1) = struct('label', 'uncorrelated antennas', 'changes', {{'channel.correlation', 'Low'}});
cases(end + 1) = struct('label', 'colored interference', 'changes', ...
    {{'noise.covariance', interference}});
cases(end + 1) = struct('label', 'lossy, late control', 'changes', ...
    {{'control.pDrop', 0.05, 'control.pLate', 0.1}});

spec = struct('type', 'maxrmac', 'theta', [1; 1.5]);
fprintf('[maxrmac ablation, weighted-rate gap to the NR oracle]\n');
fprintf('%-24s', 'case');

for m = 2:numel(base.methods)
    fprintf(' %18s', base.methods{m});
end

fprintf('\n%s\n', repmat('-', 1, 24 + 19 * (numel(base.methods) - 1)));
table = struct('label', {}, 'gap', {}, 'ci', {});

for c = 1:numel(cases)
    cfg = base;

    for k = 1:2:numel(cases(c).changes)
        parts = strsplit(cases(c).changes{k}, '.');
        cfg = setfield(cfg, parts{:}, cases(c).changes{k + 1});
    end

    evalc('[results, env] = nr_experiment(cfg, spec);');
    evalc('stats = nr_summary(results, cfg, env, {''nr-oracle''}, cases(c).label);');
    gap = stats.gaps.nr_oracle;
    table(end + 1) = struct('label', cases(c).label, 'gap', gap.mean, 'ci', gap.ci); %#ok<SAGROW>
    fprintf('%-24s', cases(c).label);

    for m = 2:numel(base.methods)
        fprintf(' %+8.2f%%+-%5.2f%%', 100 * gap.mean(m), 100 * gap.ci(m));
    end

    fprintf('\n');
end

outputDir = fullfile(root, 'experiments', 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'nrAblation.mat'), 'base', 'cases', 'table');
