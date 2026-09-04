/**
 * @file test_minpmac.cpp
 * @brief Comprehensive test suite for minPMAC MIMO C++ implementation
 *
 * Ports test cases from MATLAB test_minPMACMIMO.m and adds performance
 * benchmarks.
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

#include "minpmac.hpp"

using namespace minpmac;

struct TestResult {
    bool passed;
    std::string name;
    double elapsed_ms;
    double max_rate_deficit;
    double total_energy;
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
TestResult validate_result(const MinPMACResult& result, const VectorXd& bu_min,
                           const VectorXd& w, const std::vector<int>& Lxu,
                           const std::string& name) {
    TestResult test;
    test.passed = true;
    test.name = name;

    const int U = bu_min.size();
    const int N = result.bun.cols();

    std::cout << "  config: " << U << " users, " << N << " tones" << std::endl;

    std::cout << "  target rates: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(2) << bu_min(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  achieved rates: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.bu_a(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    double max_deficit = 0.0;
    for (int u = 0; u < U; ++u) {
        double deficit = bu_min(u) - result.bu_a(u);
        max_deficit = std::max(max_deficit, deficit);
    }
    test.max_rate_deficit = max_deficit;

    std::cout << "  rate deficit: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4)
                  << result.bu_a(u) - bu_min(u) << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    double total_energy = 0.0;
    for (int u = 0; u < U; ++u) {
        total_energy += w(u) * result.Eu_avg(u) * N;
    }
    test.total_energy = total_energy;

    std::cout << "  avg energy per user: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.Eu_avg(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  total weighted energy: " << std::fixed
              << std::setprecision(4) << total_energy << std::endl;

    std::cout << "  theta: [";
    for (int u = 0; u < U; ++u)
        std::cout << std::fixed << std::setprecision(4) << result.theta(u)
                  << (u < U - 1 ? " " : "");
    std::cout << "]" << std::endl;

    std::cout << "  feas_flag: " << result.feas_flag << std::endl;

    if (!result.orderings.empty()) {
        std::cout << "  orderings: " << result.orderings.size() << std::endl;
        for (size_t k = 0; k < result.orderings.size(); ++k) {
            std::cout << "    [" << k << "]: [";
            for (size_t i = 0; i < result.orderings[k].size(); ++i) {
                std::cout << result.orderings[k][i] + 1;

                if (i < result.orderings[k].size() - 1) std::cout << " ";
            }
            std::cout << "]";
            if (result.frac.size() > 0 &&
                k < static_cast<size_t>(result.frac.size())) {
                std::cout << " weight=" << std::fixed << std::setprecision(4)
                          << result.frac(k);
            }
            std::cout << std::endl;
        }
    }

    std::cout << "  elapsed time: " << std::fixed << std::setprecision(3)
              << result.elapsed_sec * 1000 << " ms" << std::endl;

    const double rate_tol = 1e-2;

    if (result.feas_flag == 0) {
        std::cout << "  [x] solver reported infeasible (feas_flag=0)"
                  << std::endl;
        test.passed = false;
    } else if (max_deficit > rate_tol) {
        std::cout << "  [x] rate constraints not met (max deficit: "
                  << max_deficit << ")" << std::endl;
        test.passed = false;
    } else {
        std::cout << "  [p] rate constraints satisfied" << std::endl;
    }

    bool psd_ok = true;
    for (int n = 0; n < N && psd_ok; ++n) {
        if (!is_psd(result.Rxx[n], -1e-9)) {
            psd_ok = false;
        }
    }
    if (psd_ok) {
        std::cout << "  [p] covariance matrices are PSD" << std::endl;
    } else {
        std::cout << "  [x] some covariance matrices are not PSD" << std::endl;
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

    if (test.passed) {
        std::cout << "  [p] test passed: " << name << std::endl;
    } else {
        std::cout << "  [x] test failed: " << name << std::endl;
    }

    return test;
}

TestResult test_reference_siso() {
    std::cout << "[reference] siso (2 users, 4 tones)" << std::endl;

    const int Ly = 1, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu_min(2);
    bu_min << 24, 24;
    VectorXd w(2);
    w << 1, 1;
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

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "reference siso");
}

TestResult test_reference_mimo() {
    std::cout << "[reference] mimo (2 users, 2x2, 2 tones)" << std::endl;

    const int Ly = 2, U = 2, N = 2;
    std::vector<int> Lxu = {2, 2};
    VectorXd bu_min(2);
    bu_min << 10, 10;
    VectorXd w(2);
    w << 1, 1;
    int cb = 1;

    std::vector<MatrixXcd> H(N);
    H[0].resize(Ly, 4);
    H[0] << Complex(3.2, 0), Complex(2.1, 0), Complex(2.5, 0), Complex(3.0, 0),
        Complex(1.8, 0), Complex(2.9, 0), Complex(2.2, 0), Complex(1.9, 0);
    H[1].resize(Ly, 4);
    H[1] << Complex(2.9, 0), Complex(2.5, 0), Complex(2.8, 0), Complex(2.3, 0),
        Complex(2.0, 0), Complex(3.1, 0), Complex(1.7, 0), Complex(2.6, 0);

    MinPMACConfig config;
    config.verbose = true;

    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "reference mimo");
}

TestResult test_random_siso() {
    std::cout << "[test] random siso (3 users, 4 tones)" << std::endl;

    const int Ly = 2, U = 3, N = 4;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd bu_min(3);
    bu_min << 2, 1.5, 2.5;
    VectorXd w(3);
    w << 1, 1, 1;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, U);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < U; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "random siso");
}

TestResult test_random_mimo() {
    std::cout << "[test] random mimo (3 users, mixed antennas)" << std::endl;

    const int Ly = 3, N = 6;
    std::vector<int> Lxu = {2, 3, 2};
    const int U = Lxu.size();
    int Ltot = 0;
    for (int u = 0; u < U; ++u) Ltot += Lxu[u];

    VectorXd bu_min(3);
    bu_min << 3, 4, 2.5;
    VectorXd w(3);
    w << 1, 2, 1;
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

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "random mimo");
}

TestResult test_stress() {
    std::cout << "[test] stress (6 users, 64 tones)" << std::endl;

    const int Ly = 4, N = 64;
    std::vector<int> Lxu = {3, 2, 4, 2, 1, 2};
    const int U = Lxu.size();
    int Ltot = 0;
    for (int u = 0; u < U; ++u) Ltot += Lxu[u];

    VectorXd bu_min(U);
    bu_min << 2, 1.8, 2.2, 1.5, 2, 1.8;
    VectorXd w(U);
    w << 1, 1, 1, 1, 1, 1;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, Ltot);
        MatrixXcd H_base(Ly, Ltot);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < Ltot; ++j) {
                H_base(i, j) = Complex(dist(gen), dist(gen));
            }
        }

        H[n] = H_base;
        for (int i = 0; i < std::min(Ly, Ltot); ++i) {
            H[n](i, i) += 0.1;
        }
    }

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "stress test");
}

TestResult test_real_baseband() {
    std::cout << "[test] real baseband (3 users, 4 tones)" << std::endl;

    const int Ly = 2, U = 3, N = 4;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd bu_min(3);
    bu_min << 1, 1, 1;
    VectorXd w(3);
    w << 1, 1, 1;
    int cb = 2;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, U);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < U; ++j) {
                H[n](i, j) = Complex(dist(gen), 0.0);
            }
        }
    }

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "real baseband");
}

TestResult test_unequal_weights() {
    std::cout << "[test] unequal energy weights" << std::endl;

    const int Ly = 2, U = 3, N = 4;
    std::vector<int> Lxu = {1, 1, 1};
    VectorXd bu_min(3);
    bu_min << 2, 2, 2;
    VectorXd w(3);
    w << 1, 3, 0.5;

    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, U);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < U; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "unequal weights");
}

TestResult test_low_rate() {
    std::cout << "[test] low rate targets" << std::endl;

    const int Ly = 2, U = 2, N = 4;
    std::vector<int> Lxu = {1, 1};
    VectorXd bu_min(2);
    bu_min << 0.5, 0.5;

    VectorXd w(2);
    w << 1, 1;
    int cb = 1;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(2.0));

    std::vector<MatrixXcd> H(N);
    for (int n = 0; n < N; ++n) {
        H[n].resize(Ly, U);
        for (int i = 0; i < Ly; ++i) {
            for (int j = 0; j < U; ++j) {
                H[n](i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }

    MinPMACConfig config;
    config.verbose = false;
    auto result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);
    return validate_result(result, bu_min, w, Lxu, "low rate targets");
}

int main(int argc, char* argv[]) {
    std::vector<TestResult> results;

    results.push_back(run_test("reference siso", test_reference_siso));
    std::cout << std::endl;

    results.push_back(run_test("reference mimo", test_reference_mimo));
    std::cout << std::endl;

    results.push_back(run_test("random siso", test_random_siso));
    std::cout << std::endl;

    results.push_back(run_test("random mimo", test_random_mimo));
    std::cout << std::endl;

    results.push_back(run_test("stress test", test_stress));
    std::cout << std::endl;

    results.push_back(run_test("real baseband", test_real_baseband));
    std::cout << std::endl;

    results.push_back(run_test("unequal weights", test_unequal_weights));
    std::cout << std::endl;

    results.push_back(run_test("low rate targets", test_low_rate));
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
