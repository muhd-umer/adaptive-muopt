%TEST_MAXRESMACMIMO Collection of tests for maxRESMACMIMO
%
%   Author: Muhammad Umer
%   Organization: Stanford University

clear;
clc;
rng(42);

cpp_available = exist('maxresmac_mex', 'file') == 3;

if cpp_available
    fprintf('[info] MEX implementation available\n\n');
else
    fprintf('[info] MEX not found; testing MATLAB only\n');
    fprintf('[info] Build with: cd build && cmake .. && make maxresmac_mex\n\n');
end

%% test definitions
tests = struct();

tests(1).name = 'basic siso';
H = zeros(1, 2, 4);
H(:, :, 1) = [80 60]; H(:, :, 2) = [40 30];
H(:, :, 3) = [50 50]; H(:, :, 4) = [30 40];
tests(1).H = H; tests(1).Lxu = [1 1]; tests(1).Etotal = 5;
tests(1).theta = [1; 1]; tests(1).cb = 1;

tests(2).name = 'basic mimo';
H = zeros(2, 4, 2);
H(:, 1:2, 1) = [3.2 2.1; 1.8 2.9]; H(:, 3:4, 1) = [2.5 3.0; 2.2 1.9];
H(:, 1:2, 2) = [2.9 2.5; 2.0 3.1]; H(:, 3:4, 2) = [2.8 2.3; 1.7 2.6];
tests(2).H = H; tests(2).Lxu = [2 2]; tests(2).Etotal = 10;
tests(2).theta = [1; 1]; tests(2).cb = 1;

tests(3).name = 'single tone';
H = zeros(1, 2, 1); H(:, :, 1) = [80 60];
tests(3).H = H; tests(3).Lxu = [1 1]; tests(3).Etotal = 2;
tests(3).theta = [1; 1]; tests(3).cb = 1;

tests(4).name = 'unequal weights';
H = zeros(1, 2, 4);
H(:, :, 1) = [80 60]; H(:, :, 2) = [40 30];
H(:, :, 3) = [50 50]; H(:, :, 4) = [30 40];
tests(4).H = H; tests(4).Lxu = [1 1]; tests(4).Etotal = 5;
tests(4).theta = [1; 3]; tests(4).cb = 1;

tests(5).name = 'three users';
H = zeros(2, 3, 4);
H(:, :, 1) = [5 4 3; 4 5 4]; H(:, :, 2) = [4 3 5; 3 4 5];
H(:, :, 3) = [3 5 4; 5 3 4]; H(:, :, 4) = [4 4 4; 4 4 4];
tests(5).H = H; tests(5).Lxu = [1 1 1]; tests(5).Etotal = 6;
tests(5).theta = [1; 1; 1]; tests(5).cb = 1;

tests(6).name = 'mimo unequal ant';
tests(6).H = (randn(3, 5, 2) + 1j * randn(3, 5, 2)) / sqrt(2);
tests(6).Lxu = [2 3]; tests(6).Etotal = 8;
tests(6).theta = [1; 1.5]; tests(6).cb = 1;

tests(7).name = 'real baseband';
H = zeros(1, 2, 4);
H(:, :, 1) = [5 4]; H(:, :, 2) = [3 2];
H(:, :, 3) = [2 3]; H(:, :, 4) = [4 5];
tests(7).H = H; tests(7).Lxu = [1 1]; tests(7).Etotal = 4;
tests(7).theta = [1; 1]; tests(7).cb = 2;

tests(8).name = 'stress';
tests(8).H = (randn(4, 9, 64) + 1j * randn(4, 9, 64)) / sqrt(2);
tests(8).Lxu = [2 1 2 1 2 1]; tests(8).Etotal = 30;
tests(8).theta = [1; 2; 1.5; 0.5; 1.2; 0.8]; tests(8).cb = 1;

%% run MATLAB tests
fprintf('[matlab]\n\n');

matlab_results = run_all_tests(@(H, Lxu, Etotal, theta, cb) maxRESMACMIMO(H, Lxu, Etotal, theta, cb, false), tests);

%% run C++ tests
if cpp_available
    fprintf('[cpp]\n\n');

    cpp_results = run_all_tests(@(H, Lxu, Etotal, theta, cb) maxRESMACMIMO(H, Lxu, Etotal, theta, cb, true), tests);

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
            tests(t).name, mr.weighted_rate, cr.weighted_rate, ...
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
        result = run_test(solver, tc.H, tc.Lxu, tc.Etotal, tc.theta, tc.cb, tc.name);
        results = [results; result];
        fprintf('\n');
    end

end

function result = run_test(solver, H, Lxu, Etotal, theta, cb, test_name)
    result.passed = true;
    result.elapsed_ms = 0;
    result.weighted_rate = 0;
    result.energy = 0;

    try
        tic;
        [Rxxs, Eun, w, bun] = solver(H, Lxu, Etotal, theta, cb);
        elapsed = toc;
        result.elapsed_ms = elapsed * 1000;

        theta = theta(:);
        U = length(theta);

        bu = sum(bun, 2);
        weighted_rate = sum(theta .* bu);
        result.weighted_rate = weighted_rate;
        total_energy = sum(Eun, 'all');
        result.energy = total_energy;
        energy_err = abs(total_energy - Etotal) / max(Etotal, 1e-12);

        fprintf('  config: %d users, Etotal = %.2f\n', U, Etotal);
        fprintf('  weights theta: [%s]\n', num2str(theta'));
        fprintf('  per-user rates: [%s]\n', num2str(bu'));
        fprintf('  total rate: %.4f, weighted rate: %.4f\n', sum(bu), weighted_rate);
        fprintf('  total energy used: %.6f (target: %.6f)\n', total_energy, Etotal);
        fprintf('  energy constraint error: %.2e\n', energy_err);
        fprintf('  lagrange multiplier w: [%s]\n', num2str(w'));
        fprintf('  elapsed time: %.3f ms\n', result.elapsed_ms);

        if energy_err < 0.01
            fprintf('  [p] energy constraint satisfied\n');
        else
            fprintf('  [x] energy constraint violated (error: %.2e)\n', energy_err);
            result.passed = false;
        end

        if all(bun(:) >= -1e-6)
            fprintf('  [p] rates are non-negative\n');
        else
            fprintf('  [x] some rates are negative\n');
            result.passed = false;
        end

        psd_ok = check_psd(Rxxs);

        if psd_ok
            fprintf('  [p] covariance matrices are PSD\n');
        else
            fprintf('  [x] some covariance matrices are not PSD\n');
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

function ok = check_psd(Rxxs)
    ok = true;

    if iscell(Rxxs)

        for u = 1:size(Rxxs, 1)

            for n = 1:size(Rxxs, 2)
                R = Rxxs{u, n};

                if ~isempty(R) && any(real(eig(0.5 * (R + R'))) < -1e-9)
                    ok = false;
                    return;
                end

            end

        end

    else

        for u = 1:size(Rxxs, 3)

            for n = 1:size(Rxxs, 4)
                R = Rxxs(:, :, u, n);

                if any(real(eig(0.5 * (R + R'))) < -1e-9)
                    ok = false;
                    return;
                end

            end

        end

    end

end
