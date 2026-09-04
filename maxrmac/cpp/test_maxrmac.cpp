/**
 * @file test_maxrmac.cpp
 * 
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "maxrmac.hpp"

using namespace maxrmac;

struct TestResult {
    bool passed;
    std::string name;
    double elapsed_ms;
    double energy_error;
    double weighted_rate;
};

/**
 * @brief Run a single test case
 */
template <typename Func>
TestResult run_test(const std::string& name, Func test_func) {
    TestResult result;
    result.name = name;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        result = test_func();
        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        result.name = name;
    } catch (const std::exception& e) {
        result.passed = false;
        result.elapsed_ms = 0;
        std::cout << "  [x] Test crashed: " << e.what() << std::endl;
    }

    return result;
}

/**
 * @brief Validate solver result
 */
TestResult validate_result(const MaxRMACResult& result, const VectorXd& Eu,
                           const VectorXd& theta, const std::vector<int>& Lxu,
                           const std::string& name) {
    TestResult test;
    test.passed = true;
    test.name = name;

    const int U = Eu.size();
    const int N = result.bun.cols();

    VectorXd bu = result.bun.rowwise().sum();
    VectorXd energy_used = result.Eun.rowwise().sum();

    double weighted_rate = theta.dot(bu);
    test.weighted_rate = weighted_rate;

    double max_energy_err = 0.0;
    for (int u = 0; u < U; ++u) {
        double rel_err =
            std::abs(energy_used(u) - Eu(u)) / std::max(Eu(u), 1e-12);
        max_energy_err = std::max(max_energy_err, rel_err);
    }
    test.energy_error = max_energy_err;

    std::cout << "  config: " << U << " users, Eu = [";
    for (int u = 0; u < U; ++u) std::cout << Eu(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  weights theta: [";
    for (int u = 0; u < U; ++u) std::cout << theta(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  per-user rates: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << bu(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  total rate: " << std::fixed << std::setprecision(4)
              << bu.sum() << ", weighted rate: " << weighted_rate << std::endl;

    std::cout << "  energy used: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << energy_used(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  energy constraint error: " << std::scientific
              << max_energy_err << std::endl;

    std::cout << "  lagrange multiplier w: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.w(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  elapsed time: " << std::fixed << std::setprecision(3)
              << result.elapsed_sec * 1000 << " ms" << std::endl;

    if (max_energy_err < 1e-3) {
        std::cout << "  [p] all energy constraints satisfied (max rel error: "
                  << std::scientific << max_energy_err << ")" << std::endl;
    } else {
        std::cout << "  [x] energy constraints violated (max rel error: "
                  << std::scientific << max_energy_err << ")" << std::endl;
        test.passed = false;
    }

    bool rates_ok = true;
    for (int u = 0; u < U && rates_ok; ++u) {
        for (int n = 0; n < N && rates_ok; ++n) {
            if (result.bun(u, n) < -1e-8) rates_ok = false;
        }
    }
    if (rates_ok) {
        std::cout << "  [p] rates are non-negative" << std::endl;
    } else {
        std::cout << "  [x] some rates are negative" << std::endl;
        test.passed = false;
    }

    bool w_ok = true;
    for (int u = 0; u < U; ++u) {
        if (result.w(u) < -1e-12) w_ok = false;
    }
    if (w_ok) {
        std::cout << "  [p] lagrange multipliers are non-negative" << std::endl;
    } else {
        std::cout << "  [x] some lagrange multipliers are negative"
                  << std::endl;
        test.passed = false;
    }

    bool psd_ok = true;
    for (int u = 0; u < U && psd_ok; ++u) {
        for (int n = 0; n < N && psd_ok; ++n) {
            if (!is_psd(result.Rxx[u][n], -1e-9)) {
                psd_ok = false;
            }
        }
    }
    if (psd_ok) {
        std::cout << "  [p] covariance matrices are PSD" << std::endl;
    } else {
        std::cout << "  [x] some covariance matrices are not PSD" << std::endl;
        test.passed = false;
    }

    if (test.passed) {
        std::cout << "  [p] test passed: " << name << std::endl;
    } else {
        std::cout << "  [x] test failed: " << name << std::endl;
    }

    return test;
}

TestResult test_basic_siso() {
    std::cout << "[test] basic siso (2 users, 4 tones)" << std::endl;

    const int Ly = 1, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd Eu(2);
    Eu << 2, 2;
    VectorXd theta(2);
    theta << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 2);
    H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(Ly, 2);
    H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(Ly, 2);
    H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(Ly, 2);
    H[3] << Complex(30, 0), Complex(40, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "basic siso");
}

TestResult test_basic_mimo() {
    std::cout << "[test] basic mimo (2 users, 2x2, 2 tones)" << std::endl;

    const int Ly = 2, U = 2, N = 2;
    std::vector<int> Lxu = {2, 2};
    VectorXd Eu(2);
    Eu << 5, 6;
    VectorXd theta(2);
    theta << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 4);
    H[0] << Complex(3.2, 0), Complex(2.1, 0), Complex(2.5, 0), Complex(3.0, 0),
        Complex(1.8, 0), Complex(2.9, 0), Complex(2.2, 0), Complex(1.9, 0);
    H[1].resize(Ly, 4);
    H[1] << Complex(2.9, 0), Complex(2.5, 0), Complex(2.8, 0), Complex(2.3, 0),
        Complex(2.0, 0), Complex(3.1, 0), Complex(1.7, 0), Complex(2.6, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "basic mimo");
}

TestResult test_single_tone() {
    std::cout << "[test] single tone" << std::endl;

    const int Ly = 1, U = 2, N = 1;
    std::vector<int> Lxu = {1, 1};
    VectorXd Eu(2);
    Eu << 2, 2;
    VectorXd theta(2);
    theta << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(1);
    H[0].resize(Ly, 2);
    H[0] << Complex(80, 0), Complex(60, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "single tone");
}

TestResult test_unequal_weights() {
    std::cout << "[test] unequal weights" << std::endl;

    const int Ly = 1, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd Eu(2);
    Eu << 3, 3;
    VectorXd theta(2);
    theta << 1, 3;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 2);
    H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(Ly, 2);
    H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(Ly, 2);
    H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(Ly, 2);
    H[3] << Complex(30, 0), Complex(40, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "unequal weights");
}

TestResult test_unequal_energies() {
    std::cout << "[test] unequal per-user energies" << std::endl;

    const int Ly = 1, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd Eu(2);
    Eu << 5, 2;
    VectorXd theta(2);
    theta << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 2);
    H[0] << Complex(80, 0), Complex(60, 0);
    H[1].resize(Ly, 2);
    H[1] << Complex(40, 0), Complex(30, 0);
    H[2].resize(Ly, 2);
    H[2] << Complex(50, 0), Complex(50, 0);
    H[3].resize(Ly, 2);
    H[3] << Complex(30, 0), Complex(40, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "unequal energies");
}

TestResult test_three_users() {
    std::cout << "[test] three users" << std::endl;

    const int Ly = 2, U = 3, N = 4;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd Eu(3);
    Eu << 3, 4, 3.5;
    VectorXd theta(3);
    theta << 1, 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 3);
    H[0] << Complex(5, 0), Complex(4, 0), Complex(3, 0), Complex(4, 0),
        Complex(5, 0), Complex(4, 0);
    H[1].resize(Ly, 3);
    H[1] << Complex(4, 0), Complex(3, 0), Complex(5, 0), Complex(3, 0),
        Complex(4, 0), Complex(5, 0);
    H[2].resize(Ly, 3);
    H[2] << Complex(3, 0), Complex(5, 0), Complex(4, 0), Complex(5, 0),
        Complex(3, 0), Complex(4, 0);
    H[3].resize(Ly, 3);
    H[3] << Complex(4, 0), Complex(4, 0), Complex(4, 0), Complex(4, 0),
        Complex(4, 0), Complex(4, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "three users");
}

TestResult test_mimo_mixed_antennas() {
    std::cout << "[test] mimo with unequal antennas" << std::endl;

    const int Ly = 3, U = 2, N = 2;
    std::vector<int> Lxu = {2, 3};
    VectorXd Eu(2);
    Eu << 4, 6;
    VectorXd theta(2);
    theta << 1, 1.5;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, 5);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < 5; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "mimo unequal antennas");
}

TestResult test_real_baseband() {
    std::cout << "[test] real baseband" << std::endl;

    const int Ly = 1, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd Eu(2);
    Eu << 2, 2;
    VectorXd theta(2);
    theta << 1, 1;
    int cb = 2;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 2);
    H[0] << Complex(5, 0), Complex(4, 0);
    H[1].resize(Ly, 2);
    H[1] << Complex(3, 0), Complex(2, 0);
    H[2].resize(Ly, 2);
    H[2] << Complex(2, 0), Complex(3, 0);
    H[3].resize(Ly, 2);
    H[3] << Complex(4, 0), Complex(5, 0);

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "real baseband");
}

TestResult test_uniform_antennas() {
    std::cout << "[test] uniform antennas input (scalar Lxu)" << std::endl;

    const int Ly = 2, U = 3, Lxu_scalar = 2, N = 3;
    VectorXd Eu(3);
    Eu << 3, 4, 5;
    VectorXd theta(3);
    theta << 1, 1, 1;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, U * Lxu_scalar);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < U * Lxu_scalar; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    std::vector<int> Lxu(U, Lxu_scalar);
    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "uniform antennas");
}

TestResult test_stress() {
    std::cout << "[test] stress test" << std::endl;

    const int Ly = 4, N = 64;
    std::vector<int> Lxu = {2, 1, 2, 1, 2, 1};
    const int U = Lxu.size();
    int Ltot = 0;
    for (int u = 0; u < U; ++u) Ltot += Lxu[u];

    VectorXd Eu(U);
    Eu << 5, 4, 6, 3, 5, 4;
    VectorXd theta(U);
    theta << 1, 2, 1.5, 0.5, 1.2, 0.8;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, Ltot);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < Ltot; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    auto result = maxRMACMIMO(H, Lxu, Eu, theta, cb);
    return validate_result(result, Eu, theta, Lxu, "stress test");
}

int main(int argc, char* argv[]) {
    std::vector<TestResult> results;

    results.push_back(run_test("basic siso", test_basic_siso));
    std::cout << std::endl;

    results.push_back(run_test("basic mimo", test_basic_mimo));
    std::cout << std::endl;

    results.push_back(run_test("single tone", test_single_tone));
    std::cout << std::endl;

    results.push_back(run_test("unequal weights", test_unequal_weights));
    std::cout << std::endl;

    results.push_back(run_test("unequal energies", test_unequal_energies));
    std::cout << std::endl;

    results.push_back(run_test("three users", test_three_users));
    std::cout << std::endl;

    results.push_back(
        run_test("mimo mixed antennas", test_mimo_mixed_antennas));
    std::cout << std::endl;

    results.push_back(run_test("real baseband", test_real_baseband));
    std::cout << std::endl;

    results.push_back(run_test("uniform antennas", test_uniform_antennas));
    std::cout << std::endl;

    results.push_back(run_test("stress test", test_stress));
    std::cout << std::endl;

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed)
            ++passed;
        else
            ++failed;
    }

    std::cout << std::endl;
    std::cout << "[passed] " << passed << "/" << results.size() << std::endl;
    std::cout << "[failed] " << failed << "/" << results.size() << std::endl;
    std::cout << std::endl;

    std::cout << "[summary]" << std::endl;
    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(25) << r.name << ": "
                  << std::fixed << std::setprecision(3) << r.elapsed_ms << " ms"
                  << std::endl;
    }

    return failed > 0 ? 1 : 0;
}
