%TEST_MINPMAC Test suite for non-CVX minPMAC implementations

clear;
clc;
cvx_clear;
rng(42);

%% reference tests

fprintf('[reference] SISO\n');
H_siso = zeros(1, 2, 4);
H_siso(:, :, 1) = [80 60];
H_siso(:, :, 2) = [40 30];
H_siso(:, :, 3) = [50 50];
H_siso(:, :, 4) = [30 40];
Lx_siso = [1 1];
bu_ref_siso = [24 24];
w_ref = [1 1];
cb = 1;

[flag, bu_a, info] = minPMACMIMO(H_siso, Lx_siso, bu_ref_siso, w_ref, cb);
total_energy = sum(info.Rxx{1}, 'all');

fprintf('  users: %d, tones: %d, rx: %d\n', length(Lx_siso), size(H_siso, 3), size(H_siso, 1));
fprintf('  target rates: [%s]\n', num2str(bu_ref_siso));
fprintf('  achieved rates: [%s]\n', num2str(bu_a));
fprintf('  feasibility flag: %d\n', flag);
fprintf('  total energy: %.4f (expected 6.4746, diff %.6f)\n', total_energy, abs(total_energy - 6.4746));
fprintf('  order: [%s]\n', num2str(info.orderings{1}));
fprintf('\n');

fprintf('[reference] MIMO\n');
Ly_ref = 2;
Lx_ref = [2 2];
N_ref = 2;
H_mimo = zeros(Ly_ref, sum(Lx_ref), N_ref);
H_mimo(:, 1:2, 1) = [3.2 2.1; 1.8 2.9];
H_mimo(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H_mimo(:, 1:2, 2) = [2.9 2.5; 2.0 3.1];
H_mimo(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
bu_ref_mimo = [10 10];
w_ref_mimo = [1 1];

[flag, bu_a, info] = minPMACMIMO(H_mimo, Lx_ref, bu_ref_mimo, w_ref_mimo, cb);

fprintf('  users: %d, tones: %d, rx: %d\n', length(Lx_ref), N_ref, Ly_ref);
fprintf('  target rates: [%s]\n', num2str(bu_ref_mimo));
fprintf('  achieved rates: [%s]\n', num2str(bu_a));
fprintf('  feasibility flag: %d\n', flag);
fprintf('  total energy: %.4f\n', sum(info.Rxx{1}, 'all'));
fprintf('  order: [%s]\n', num2str(info.orderings{1}));
fprintf('\n');

%% randomized and edge-case tests

fprintf('[test] SISO\n');
Ly = 2;
U = 3;
N = 4;
H_siso_rand = (randn(Ly, U, N) + 1j * randn(Ly, U, N)) / sqrt(2);
Lx = ones(1, U);
bu_min = [2, 1.5, 2.5];
w = [1, 1, 1];
cb = 1;

run_test(@() minPMACMIMO(H_siso_rand, Lx, bu_min, w, cb), bu_min);

fprintf('[test] MIMO\n');
Ly = 3;
Lx = [2, 3, 2];
U = length(Lx);
Ltot = sum(Lx);
N = 6;
H_mimo_rand = (randn(Ly, Ltot, N) + 1j * randn(Ly, Ltot, N)) / sqrt(2);
bu_min = [3, 4, 2.5];
w = [1, 2, 1];

run_test(@() minPMACMIMO(H_mimo_rand, Lx, bu_min, w, cb), bu_min);

fprintf('[test] Stress\n');
Ly = 4;
Lx = [3, 2, 4, 2, 1, 2];
U = length(Lx);
Ltot = sum(Lx);
N = 8;
H_large = zeros(Ly, Ltot, N);

for n = 1:N
    H_base = (randn(Ly, Ltot) + 1j * randn(Ly, Ltot)) / sqrt(2);
    H_large(:, :, n) = H_base + 0.1 * eye(Ly, Ltot);
end

bu_min = [2, 1.8, 2.2, 1.5, 2, 1.8];
w = ones(1, U);

run_test(@() minPMACMIMO(H_large, Lx, bu_min, w, cb), bu_min);

fprintf('[test] Real Baseband\n');
H_real = randn(2, 3, 4);
Lx = ones(1, 3);
bu_min = [1, 1, 1];
w = [1, 1, 1];
cb = 2;

run_test(@() minPMACMIMO(H_real, Lx, bu_min, w, cb), bu_min);

fprintf('[test] Infeasible\n');
Ly = 2;
U = 3;
N = 1;
h_base = [1; 0.5];
H_infeas = zeros(Ly, U, N);
H_infeas(:, 1, 1) = 0.01 * h_base;
H_infeas(:, 2, 1) = 0.01 * h_base;
H_infeas(:, 3, 1) = 0.01 * h_base;
Lx = ones(1, U);
bu_min = [8, 8, 8];
w = [1, 1, 1];
cb = 1;

run_test(@() minPMACMIMO(H_infeas, Lx, bu_min, w, cb), bu_min, true);

%% helper functions

function run_test(solver, target_rates, expect_infeasible)

    if nargin < 3
        expect_infeasible = false;
    end

    try
        tic;
        [flag, bu_a, info] = solver();
        elapsed = toc;

        if isfield(info, 'clusters')
            num_clusters = length(info.clusters);
        else
            num_clusters = 0;
        end

        if isfield(info, 'orderings')
            num_orders = length(info.orderings);
        else
            num_orders = 0;
        end

        fprintf('  target rates: [%s]\n', num2str(target_rates));
        fprintf('  achieved rates: [%s]\n', num2str(bu_a));
        fprintf('  rate error: [%s]\n', num2str(bu_a - target_rates));
        fprintf('  feasibility flag: %d\n', flag);
        fprintf('  solution status: %s\n', info.sol_status);
        fprintf('  clusters: %d, orders: %d, elapsed: %.3f s\n', ...
            num_clusters, num_orders, elapsed);

        if expect_infeasible

            if flag == 0
                fprintf('  [p] infeasible case detected\n');
            else
                fprintf('  [x] infeasible case not detected\n');
            end

        else

            if flag > 0
                fprintf('  [p] test passed\n');
            else
                fprintf('  [x] test failed (infeasible)\n');
            end

        end

    catch ME
        fprintf('  [x] test crashed: %s\n', ME.message);
    end

    fprintf('\n');
end
