%ADA_NRPUSCH_MAXRESMAC Pilot-driven maxRESMAC on an NR uplink with a shared energy budget
%
% The UEs share an aggregate energy budget below the sum of their own
% power caps, so the experiment keeps maxRESMAC's coupled constraint while
% every deployed action also respects each UE's cap. Unrestricted canonical
% maxRESMAC is the theoretical upper bound, the capped canonical design is
% the physically constrained continuous reference, and the perfect-CSI NR
% oracle bounds what the codebook and TPC can realize. ue-pilot runs the
% canonical solver on the UEs through reciprocity (TDD, non-codebook PUSCH)
% with a flat power spectral density; ue-pilot-free lifts that restriction

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'ada_utils')), fullfile(root, 'maxresmac'), ...
    fullfile(root, 'maxrmac'));

cfg = nr_default_config();
cfg.methods = {'canonical', 'canonical-ue', 'nr-oracle', 'nr-csi', 'nr-pilot', ...
    'cov-pilot', 'ue-pilot', 'ue-pilot-free', 'hold'};

if strcmp(getenv('ADA_QUICK'), '1')
    cfg.realizations = 4;
    cfg.horizonSlots = 200;
    cfg.warmupSlots = 50;
end

% the capped reference costs a maxRMAC budget-split bisection every slot
if strcmp(getenv('ADA_SKIP_CAPPED'), '1')
    cfg.methods = setdiff(cfg.methods, {'canonical-ue'}, 'stable');
end

spec = struct('type', 'maxresmac', 'theta', [1; 1.5], 'sumFraction', 0.75);
tstart = tic;
[results, env] = nr_experiment(cfg, spec);
fprintf('[maxresmac] %d realizations in %.1f s\n', cfg.realizations, toc(tstart));
references = intersect({'canonical', 'canonical-ue', 'nr-oracle'}, cfg.methods, 'stable');
stats = nr_summary(results, cfg, env, references, ...
    'pilot-driven maxRESMAC on NR PUSCH');

outputDir = fullfile(root, 'experiments', 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'nrPUSCH_maxRESMAC.mat'), 'cfg', 'spec', 'stats');
