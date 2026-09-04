%TEST_MINPMACMIMO Collection of tests for minPMACMIMO
%
%   Author: Muhammad Umer
%   Organization: Stanford University

clear;
clc;
rng(42);

cpp_available = exist('minpmac_mex', 'file') == 3;

if cpp_available
    fprintf('[info] MEX implementation available\n\n');
else
    fprintf('[info] MEX not found; testing MATLAB only\n');
    fprintf('[info] Build with: cd build && cmake .. && make minpmac_mex\n\n');
end

%% test definitions
tests = struct();

tests(1).name = 'reference siso';
H = zeros(1, 2, 4);
H(:, :, 1) = [80 60]; H(:, :, 2) = [40 30];
H(:, :, 3) = [50 50]; H(:, :, 4) = [30 40];
tests(1).H = H; tests(1).Lx = [1 1]; tests(1).bu_min = [24 24];
tests(1).w = [1 1]; tests(1).cb = 1;

tests(2).name = 'reference mimo';
H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9]; H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1]; H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(2).H = H; tests(2).Lx = [2 2]; tests(2).bu_min = [10 10];
tests(2).w = [1 1]; tests(2).cb = 1;

tests(3).name = 'unequal weights';
H = zeros(1, 2, 4);
H(:, :, 1) = [80 60]; H(:, :, 2) = [40 30];
H(:, :, 3) = [50 50]; H(:, :, 4) = [30 40];
tests(3).H = H; tests(3).Lx = [1 1]; tests(3).bu_min = [20 20];
tests(3).w = [1 2]; tests(3).cb = 1;

tests(4).name = 'mimo unequal ant';
tests(4).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(4).Lx = [2 3]; tests(4).bu_min = [4 6];
tests(4).w = [1 1.5]; tests(4).cb = 1;

tests(5).name = 'real baseband';
H = zeros(1, 2, 4);
H(:, :, 1) = [5 4]; H(:, :, 2) = [3 2];
H(:, :, 3) = [2 3]; H(:, :, 4) = [4 5];
tests(5).H = H; tests(5).Lx = [1 1]; tests(5).bu_min = [2 2];
tests(5).w = [1 1]; tests(5).cb = 2;

tests(6).name = 'uniform Lx';
tests(6).H = (randn(2, 6, 3) + 1j * randn(2, 6, 3)) / sqrt(2);
tests(6).Lx = 2; tests(6).bu_min = [3 4 5];
tests(6).w = [1 1 1]; tests(6).cb = 1;

tests(7).name = 'stress';
tests(7).H = (randn(4, 9, 64) + 1j * randn(4, 9, 64)) / sqrt(2);
tests(7).Lx = [2 1 2 1 2 1]; tests(7).bu_min = [2 1.8 2.2 1.5 2 1.8];
tests(7).w = [1 2 1.5 0.5 1.2 0.8]; tests(7).cb = 1;

%% run MATLAB tests
fprintf('[matlab]\n\n');

matlab_results = run_all_tests(@(H, Lx, bu_min, w, cb) minPMACMIMO(H, Lx, bu_min, w, cb, false), tests);

%% run C++ tests
if cpp_available
    fprintf('[cpp]\n\n');

    cpp_results = run_all_tests(@(H, Lx, bu_min, w, cb) minPMACMIMO(H, Lx, bu_min, w, cb, true), tests);

    %% comparison table
    fprintf('[comparison]\n');

    fprintf('%-18s | %10s %10s | %10s %10s | %8s %8s | %7s\n', ...
        'test', 'rate[.m]', 'rate[.cpp]', 'energy[.m]', 'energy[.cpp]', 'ms[.m]', 'ms[.cpp]', 'speedup');
    fprintf('%s\n', repmat('-', 1, 100));

    for t = 1:length(tests)
        mr = matlab_results(t);
        cr = cpp_results(t);
        speedup = mr.elapsed_ms / max(cr.elapsed_ms, 0.01);
        fprintf('%-18s | %10.4f %10.4f | %11.4f %11.4f | %8.2f %8.2f | %6.1fx\n', ...
            tests(t).name, mr.total_rate, cr.total_rate, ...
            mr.energy, cr.energy, mr.elapsed_ms, cr.elapsed_ms, speedup);
    end

    fprintf('\n');
end

%% helper functions

function results = run_all_tests(solver, tests)
    results = [];

    for t = 1:length(tests)
        tc = tests(t);
        fprintf('[test] %s\n', tc.name);
        result = run_test(solver, tc.H, tc.Lx, tc.bu_min, tc.w, tc.cb, tc.name);
        results = [results; result];
        fprintf('\n');
    end

end

function result = run_test(solver, H, Lx, bu_min, w, cb, test_name)
    result.passed = true;
    result.elapsed_ms = 0;
    result.total_rate = 0;
    result.energy = 0;

    try
        tic;
        [flag, bu_a, info] = solver(H, Lx, bu_min, w, cb);
        elapsed = toc;
        result.elapsed_ms = elapsed * 1000;

        bu_min = bu_min(:)';
        U = length(bu_min);
        N = size(H, 3);

        result.total_rate = sum(bu_a);
        result.energy = sum(info.Eu_avg) * N;
        rate_err = max(abs(bu_a - bu_min) ./ max(bu_min, 1e-12));

        fprintf('  config: %d users, target rates = [%s]\n', U, num2str(bu_min, '%.4g '));
        fprintf('  weights w: [%s]\n', num2str(w));
        fprintf('  achieved rates: [%s]\n', num2str(bu_a));
        fprintf('  total rate: %.4f\n', sum(bu_a));
        fprintf('  average energy: [%s]\n', num2str(info.Eu_avg));
        fprintf('  total energy: %.4f\n', result.energy);
        fprintf('  feasibility flag: %d\n', flag);
        fprintf('  rate constraint error: %.2e\n', rate_err);
        fprintf('  elapsed time: %.3f ms\n', result.elapsed_ms);

        if flag > 0
            fprintf('  [p] problem is feasible (flag=%d)\n', flag);
        else
            fprintf('  [x] problem is infeasible\n');
            result.passed = false;
        end

        if all(bu_a >= bu_min -1e-3)
            fprintf('  [p] all rate constraints satisfied\n');
        else
            fprintf('  [x] rate constraints violated\n');
            result.passed = false;
        end

        if result.passed
            fprintf('  [p] test passed: %s\n', test_name);
        else
            fprintf('  [x] test failed: %s\n', test_name);
        end

    catch ME
        result.passed = false;
        fprintf('  [x] test crashed: %s\n', ME.message);

        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end

    end

end
