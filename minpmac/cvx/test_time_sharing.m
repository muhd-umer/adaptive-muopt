% TEST_TIME_SHARING Test minPMAC time-sharing
%
% This script programmatically creates a symmetric channel and rate request
% to guarantee that a time-sharing scenario is triggered. It is designed to
% verify that the solver correctly handles cases with multiple decoding orders.

clear;
clc;
cvx_clear;
rng(100);

%% system configuration
fprintf('Creating a symmetric scenario to test time-sharing...\n');

Ly = 4;
U = 3;
N = 8;
Lx = [1, 1, 1];
Ltot = sum(Lx);
cb = 1;

%% data generation
H_t = (randn(Ly, Ltot, N) + 1j * randn(Ly, Ltot, N)) / sqrt(2);
idx_end = cumsum(Lx);
idx_start = [1, idx_end(1:end - 1) + 1];
ant_idx_u1 = idx_start(1):idx_end(1);
ant_idx_u2 = idx_start(2):idx_end(2);

H_t(:, ant_idx_u2, :) = H_t(:, ant_idx_u1, :);
fprintf('  - Channels for User 1 and User 2 have been made identical.\n');

bu_min_target = [15, 15, 20];
fprintf('  - Target rates for User 1 and User 2 are identical: [%s]\n\n', num2str(bu_min_target));

energy_weights = ones(1, U);

%% run minPMAC Test
fprintf('::: Running Symmetric Time-Sharing :::\n');

try
    [flag, bu_a, info] = minPMACMIMO_cvx(H_t, Lx, bu_min_target, energy_weights, cb);

    fprintf('\n::: Results :::\n');
    fprintf('FEAS_FLAG: %d\n', flag);
    fprintf('Sol. status: %s\n', info.sol_status);

    if isfield(info, 'clusters')
        fprintf('Number of clusters found: %d\n', length(info.clusters));
        fprintf('Cluster details: ');

        for i = 1:length(info.clusters)
            fprintf('{ %s } ', num2str(info.clusters{i}));
        end

        fprintf('\n');
    end

    if isfield(info, 'orderings')
        fprintf('Number of decoding orders found: %d\n', length(info.orderings));

        if ~isempty(info.orderings)

            for o = 1:length(info.orderings)
                fprintf('  Order p=%d: [%s]\n', o, num2str(info.orderings{o}));
            end

        end

    end

    fprintf('\n::: Outcome :::\n');

    if flag == 2
        fprintf('[p] SUCCESS: A FEASIBLE time-sharing solution was found (FEAS_FLAG = 2).\n');
        fprintf('    This confirms the code correctly handled the symmetric case with multiple orders.\n');
    elseif flag == 1
        fprintf('[!] WARNING: A single-order solution was found.\n');
        fprintf('    This is unexpected for a symmetric setup. Check the `theta` values in the info struct.\n');
    else % flag == 0
        fprintf('[x] INFEASIBLE solution returned (FEAS_FLAG = 0).\n');

        if isfield(info, 'time_sharing_failed') && info.time_sharing_failed
            fprintf('    The failure occurred in the time-sharing subproblem.\n');
        else
            fprintf('    The infeasibility may be due to the main CVX problem.\n');
        end

    end

    if flag > 0
        fprintf('\n::: Rate Comparison :::\n');
        fprintf('         Target   Achieved  Difference\n');

        for u = 1:U
            fprintf('User %2d:  %7.3f  %7.3f   %+7.3f\n', u, bu_min_target(u), bu_a(u), bu_a(u) - bu_min_target(u));
        end

        fprintf('\n::: Energy Analysis :::\n');
        fprintf('Average Energy: [%s]\n', sprintf('%.4f ', info.Eu_avg));

    end

catch ME
    warning('\nAn error occurred during the script execution.\n');
    rethrow(ME);
end
