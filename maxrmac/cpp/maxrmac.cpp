/**
 * @file maxrmac_mimo.cpp
 * @brief Main maxRMAC MIMO solver implementation
 *
 * Uses ellipsoid method for dual optimization with OpenMP-parallelized
 * per-tone L-BFGS optimization.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "maxrmac.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include "solve_tone.hpp"

namespace maxrmac {

namespace {

/**
 * @brief Initialize ellipsoid for dual optimization
 */
void init_ellipsoid(const std::vector<MatrixXcd>& H, const VectorXd& Eu,
                    const VectorXd& theta, const std::vector<int>& Lxu,
                    const std::vector<int>& idx_start,
                    const std::vector<int>& idx_end, MatrixXd& A,
                    VectorXd& w_init) {
    const int N = H.size();
    const int U = Lxu.size();

    VectorXd w_max(U);

    for (int u = 0; u < U; ++u) {
        double total_gain = 0.0;

        for (int n = 0; n < N; ++n) {
            MatrixXcd Hu =
                H[n].middleCols(idx_start[u], idx_end[u] - idx_start[u] + 1);

            if (Lxu[u] == 1) {
                total_gain += std::real((Hu.adjoint() * Hu)(0, 0));
            } else {
                total_gain += Hu.squaredNorm();
            }
        }

        if (Eu(u) > 1e-12 && total_gain > 0) {
            double avg_gain = total_gain / N;
            w_max(u) = theta(u) * avg_gain / (1.0 + avg_gain * Eu(u) / N);
        } else {
            w_max(u) = theta(u);
        }
    }

    for (int u = 0; u < U; ++u) {
        w_max(u) = std::max(w_max(u), 0.1);
    }

    w_init = w_max / 2.0;
    double radius_sq = w_max.squaredNorm() / 4.0;
    A = std::max(radius_sq, 1.0) * MatrixXd::Identity(U, U);
}

/**
 * @brief Evaluate Lagrangian across all tones (parallelized)
 */
void eval_lagfunc(const std::vector<MatrixXcd>& H, const VectorXd& theta,
                  const VectorXd& w, const std::vector<int>& Lxu,
                  const std::vector<int>& idx_start,
                  const std::vector<int>& idx_end,
                  std::vector<LBFGSState>& states,
                  std::vector<ToneWorkspace>& workspaces, double& f,
                  MatrixXd& bun, MatrixXd& Eun,
                  std::vector<std::vector<MatrixXcd>>& Rxx_all) {
    const int N = H.size();
    const int U = Lxu.size();

    bun.resize(U, N);
    Eun.resize(U, N);
    Rxx_all.resize(U);
    for (int u = 0; u < U; ++u) {
        Rxx_all[u].resize(N);
    }

    std::vector<ToneResult> results(N);

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        results[n] = solve_tone(H[n], theta, w, Lxu, idx_start, idx_end,
                                states[n], workspaces[n]);
    }

    f = 0.0;
    for (int n = 0; n < N; ++n) {
        f += results[n].f;
        for (int u = 0; u < U; ++u) {
            bun(u, n) = results[n].b(u);
            Eun(u, n) = results[n].E(u);
            Rxx_all[u][n] = results[n].Rxx[u];
        }
    }
}

/**
 * @brief Compute per-user per-tone rates given covariances
 */
void compute_rates(const std::vector<MatrixXcd>& H,
                   const std::vector<std::vector<MatrixXcd>>& Rxx,
                   const VectorXd& theta, const std::vector<int>& Lxu,
                   const std::vector<int>& idx_start,
                   const std::vector<int>& idx_end, MatrixXd& bun) {
    const int N = H.size();
    const int U = Lxu.size();
    const int Ly = H[0].rows();

    std::vector<int> order(U);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return theta(a) > theta(b); });

    bun.resize(U, N);

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        MatrixXcd S_prev = MatrixXcd::Identity(Ly, Ly);
        double logdet_prev = 0.0;

        for (int pos = 0; pos < U; ++pos) {
            int u = order[pos];
            MatrixXcd Hu =
                H[n].middleCols(idx_start[u], idx_end[u] - idx_start[u] + 1);
            MatrixXcd HRH = Hu * Rxx[u][n] * Hu.adjoint();
            HRH = make_hermitian(HRH);

            MatrixXcd S_curr = S_prev + HRH;
            S_curr = make_hermitian(S_curr);

            double logdet_curr = logdet_spd(S_curr);
            bun(u, n) = logdet_curr - logdet_prev;

            S_prev = S_curr;
            logdet_prev = logdet_curr;
        }
    }
}

}  // namespace

MaxRMACResult maxRMACMIMO(const std::vector<MatrixXcd>& H,
                          const std::vector<int>& Lxu, const VectorXd& Eu,
                          const VectorXd& theta, int cb,
                          const MaxRMACConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int Ly = H[0].rows();
    const int Ltot = H[0].cols();
    int N_in = H.size();
    const int U = Lxu.size();

    if (Eu.size() != U) {
        throw std::invalid_argument("Eu must have length U");
    }
    if (theta.size() != U) {
        throw std::invalid_argument("theta must have length U");
    }
    if (cb != 1 && cb != 2) {
        throw std::invalid_argument("cb must be 1 (complex) or 2 (real)");
    }

    int Lxu_sum = 0;
    for (int u = 0; u < U; ++u) {
        Lxu_sum += Lxu[u];
    }
    if (Lxu_sum != Ltot) {
        throw std::invalid_argument("sum(Lxu) must equal H column count");
    }

    std::vector<int> idx_start(U), idx_end(U);
    idx_end[0] = Lxu[0] - 1;
    idx_start[0] = 0;
    for (int u = 1; u < U; ++u) {
        idx_start[u] = idx_end[u - 1] + 1;
        idx_end[u] = idx_start[u] + Lxu[u] - 1;
    }

    std::vector<MatrixXcd> H_work = H;
    VectorXd Eu_work = Eu;
    int N = N_in;
    bool single_tone_flag = (N_in == 1);
    double scale_single = (3.0 - cb);

    if (N_in > 1 && cb == 2) {
        if (N_in % 2 != 0) {
            throw std::invalid_argument("for cb=2 and N>1, N must be even");
        }
        H_work.resize(N_in / 2);
        N = N_in / 2;
    } else if (single_tone_flag) {
        Eu_work = Eu / scale_single;
    }

    int num_threads = config.num_threads;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    std::vector<LBFGSState> states(N);
    std::vector<ToneWorkspace> workspaces(N);

    int max_Lxu = *std::max_element(Lxu.begin(), Lxu.end());
    int total_params = 0;
    for (int u = 0; u < U; ++u) {
        total_params += 2 * Lxu[u] * Lxu[u];
    }

    for (int n = 0; n < N; ++n) {
        workspaces[n].resize(U, Ly, max_Lxu, total_params);
    }

    MatrixXd A;
    VectorXd w;
    init_ellipsoid(H_work, Eu_work, theta, Lxu, idx_start, idx_end, A, w);

    double f;
    MatrixXd bun_nats, Eun;
    std::vector<std::vector<MatrixXcd>> Rxx_all;

    int iter = 0;

    const double inv_U_plus_1 = 1.0 / (U + 1);
    const double U_sq = static_cast<double>(U * U);
    const double scale_A = U_sq / (U_sq - 1);
    const double scale_A_gt = 2.0 / (U + 1);

    while (true) {
        ++iter;

        eval_lagfunc(H_work, theta, w, Lxu, idx_start, idx_end, states,
                     workspaces, f, bun_nats, Eun, Rxx_all);

        VectorXd g = Eu_work - Eun.rowwise().sum();

        double gAg = g.transpose() * A * g;

        if (std::sqrt(gAg) <= config.err_tol) {
            break;
        }
        if (iter > config.max_iter) {
            break;
        }

        double inv_sqrt_gAg = 1.0 / std::sqrt(gAg);
        VectorXd gt = g * inv_sqrt_gAg;
        VectorXd Agt = A * gt;
        w -= inv_U_plus_1 * Agt;
        A = scale_A * (A - scale_A_gt * Agt * gt.transpose() * A);

        bool has_negative = true;
        while (has_negative) {
            has_negative = false;
            for (int u = 0; u < U; ++u) {
                if (w(u) < 0) {
                    VectorXd g_proj = VectorXd::Zero(U);
                    g_proj(u) = -1.0;
                    double gAg_proj = g_proj.transpose() * A * g_proj;

                    double inv_sqrt_gAg_proj = 1.0 / std::sqrt(gAg_proj);
                    VectorXd gt_proj = g_proj * inv_sqrt_gAg_proj;
                    VectorXd Agt_proj = A * gt_proj;
                    w -= inv_U_plus_1 * Agt_proj;
                    A = scale_A *
                        (A - scale_A_gt * Agt_proj * gt_proj.transpose() * A);
                    has_negative = true;
                    break;
                }
            }
        }
    }

    eval_lagfunc(H_work, theta, w, Lxu, idx_start, idx_end, states, workspaces,
                 f, bun_nats, Eun, Rxx_all);

    VectorXd E_used = Eun.rowwise().sum();

    for (int u = 0; u < U; ++u) {
        if (E_used(u) > 1e-12) {
            double scale_u = Eu_work(u) / E_used(u);
            for (int n = 0; n < N; ++n) {
                Rxx_all[u][n] *= scale_u;
                Eun(u, n) *= scale_u;
            }
        } else if (Eu_work(u) > 1e-12) {
            int Lu = Lxu[u];
            double energy_per_tone = Eu_work(u) / N / Lu;
            for (int n = 0; n < N; ++n) {
                Rxx_all[u][n] = energy_per_tone * MatrixXcd::Identity(Lu, Lu);
                Eun(u, n) = Eu_work(u) / N;
            }
        }
    }

    if (single_tone_flag) {
        Eun *= scale_single;
        for (int u = 0; u < U; ++u) {
            Rxx_all[u][0] *= scale_single;
        }
    }

    compute_rates(H_work, Rxx_all, theta, Lxu, idx_start, idx_end, bun_nats);

    MatrixXd bun = bun_nats / (std::log(2.0) * cb);

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double>(end_time - start_time).count();

    MaxRMACResult result;
    result.Rxx = std::move(Rxx_all);
    result.Eun = Eun;
    result.w = w;
    result.bun = bun;
    result.weighted_rate = (theta.transpose() * bun.rowwise().sum())(0);
    result.iterations = iter;
    result.elapsed_sec = elapsed;

    return result;
}

MaxRMACResult maxRMACMIMO(const std::vector<MatrixXcd>& H, int Lxu_scalar,
                          int U, const VectorXd& Eu, const VectorXd& theta,
                          int cb, const MaxRMACConfig& config) {
    std::vector<int> Lxu(U, Lxu_scalar);
    return maxRMACMIMO(H, Lxu, Eu, theta, cb, config);
}

}  // namespace maxrmac
