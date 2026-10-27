%ADA_NRPUSCH_MAXRMAC Pilot-driven maxRMAC on an NR uplink with per-UE power limits
%
% maxRMAC is the practical uplink rate formulation: each UE owns its energy
% budget, here its full-power cap spread over the allocation. The main run
% uses asymmetric gains and weights; a second run reschedules the second UE
% mid-horizon to exercise zero-budget, inactive, and returning users.
% ue-pilot runs the canonical solver on the UEs through reciprocity (TDD,
% non-codebook PUSCH) with a flat power spectral density; ue-pilot-free
% lifts that restriction

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'ada_utils')), fullfile(root, 'maxresmac'), ...
    fullfile(root, 'maxrmac'));

cfg = nr_default_config();
cfg.methods = {'canonical', 'nr-oracle', 'nr-csi', 'nr-pilot', 'ue-pilot', ...
    'ue-pilot-free', 'hold'};

if strcmp(getenv('ADA_QUICK'), '1')
    cfg.realizations = 4;
    cfg.horizonSlots = 200;
    cfg.warmupSlots = 50;
end

spec = struct('type', 'maxrmac', 'theta', cfg.theta);
tstart = tic;
[results, env] = nr_experiment(cfg, spec);
fprintf('[maxrmac] %d realizations in %.1f s\n', cfg.realizations, toc(tstart));
stats = nr_summary(results, cfg, env, {'canonical', 'nr-oracle'}, ...
    'pilot-driven maxRMAC on NR PUSCH');

rescheduled = cfg;
rescheduled.realizations = max(4, round(cfg.realizations / 5));
rescheduled.activity = true(env.U, cfg.horizonSlots);
pause = round(cfg.horizonSlots * [0.45, 0.65]);
rescheduled.activity(2, pause(1):pause(2)) = false;
tstart = tic;
[resultsActivity, env] = nr_experiment(rescheduled, spec);
fprintf('[maxrmac rescheduling] %d realizations in %.1f s, UE 2 off in slots %d-%d\n', ...
    rescheduled.realizations, toc(tstart), pause(1) - 1, pause(2) - 1);
statsActivity = nr_summary(resultsActivity, rescheduled, env, {'canonical', 'nr-oracle'}, ...
    'pilot-driven maxRMAC with a rescheduled UE');

outputDir = fullfile(root, 'experiments', 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'nrPUSCH_maxRMAC.mat'), 'cfg', 'spec', 'stats', ...
    'rescheduled', 'statsActivity');
