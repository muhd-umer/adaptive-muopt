/**
 * @file maxresmac.cpp
 * @brief maxRESMAC MIMO solver -- direct port of maxRESMACMIMO.m
 *
 * Bisection on lambda with OpenMP-parallelized per-tone L-BFGS.
 * Self-contained: uses own solve_tone and linalg_utils.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "maxresmac.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "solve_tone.hpp"

namespace maxresmac {

namespace {

/**
 * @brief Evaluate Lagrangian across all tones (parallelized)
 *
 * Mirrors eval_lagfunc in maxRESMACMIMO.m:
 *   for n = 1:N
 *       [Rxx_n, E_n] = solve_tone(H(:,:,n), ...)
 */
void eval_lagfunc(const std::vector<MatrixXcd>& H, const std::vector<int>& Lxu,
                  const std::vector<int>& idx_start,
                  const std::vector<int>& idx_end, const VectorXd& delta,
                  double lambda, MatrixXd& Eun,
                  std::vector<std::vector<MatrixXcd>>& Rxx_all) {
    const int N = H.size();
    const int U = Lxu.size();

    Eun.resize(U, N);
    Rxx_all.resize(U);
    for (int u = 0; u < U; ++u) {
        Rxx_all[u].resize(N);
    }

    std::vector<ToneResult> results(N);

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        results[n] = solve_tone(H[n], Lxu, idx_start, idx_end, delta, lambda);
    }

    for (int n = 0; n < N; ++n) {
        for (int u = 0; u < U; ++u) {
            Eun(u, n) = results[n].E(u);
            Rxx_all[u][n] = results[n].Rxx[u];
        }
    }
}

/**
 * @brief Compute per-user per-tone rates from covariance matrices
 *
 * Mirrors compute_rates in maxRESMACMIMO.m.
 * Decoding order: already sorted (passed in sorted order).
 */
void compute_rates(const std::vector<MatrixXcd>& H,
                   const std::vector<std::vector<MatrixXcd>>& Rxx,
                   const std::vector<int>& Lxu,
                   const std::vector<int>& idx_start,
                   const std::vector<int>& idx_end, int cb, MatrixXd& bun) {
    const int N = H.size();
    const int U = Lxu.size();
    const int Ly = H[0].rows();
    const double inv_log2_cb = 1.0 / (std::log(2.0) * cb);

    bun.resize(U, N);

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        MatrixXcd S_prev = MatrixXcd::Identity(Ly, Ly);

        for (int u = 0; u < U; ++u) {
            MatrixXcd Hu =
                H[n].middleCols(idx_start[u], idx_end[u] - idx_start[u] + 1);
            MatrixXcd HRH = Hu * Rxx[u][n] * Hu.adjoint();
            HRH = make_hermitian(HRH);

            MatrixXcd S_curr = S_prev + HRH;
            S_curr = make_hermitian(S_curr);

            double logdet_prev = logdet_spd(S_prev);
            double logdet_curr = logdet_spd(S_curr);

            bun(u, n) = (logdet_curr - logdet_prev) * inv_log2_cb;
            S_prev = S_curr;
        }
    }
}

}  // namespace

MaxRESMACResult maxRESMACMIMO(const std::vector<MatrixXcd>& H,
                              const std::vector<int>& Lxu, double Etotal,
                              const VectorXd& theta, int cb,
                              const MaxRESMACConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int Ly = H[0].rows();
    const int Ltot = H[0].cols();
    int N_in = H.size();
    const int U = Lxu.size();

    if (theta.size() != U) {
        throw std::invalid_argument("theta must have length U");
    }
    if (cb != 1 && cb != 2) {
        throw std::invalid_argument("cb must be 1 (complex) or 2 (real)");
    }

    int Lxu_sum = 0;
    for (int u = 0; u < U; ++u) Lxu_sum += Lxu[u];
    if (Lxu_sum != Ltot) {
        throw std::invalid_argument("sum(Lxu) must equal H column count");
    }

    // handle real baseband: take first N/2 tones
    std::vector<MatrixXcd> H_work = H;
    int N = N_in;

    if (N_in > 1 && cb == 2) {
        if (N_in % 2 != 0) {
            throw std::invalid_argument("for cb=2 and N>1, N must be even");
        }
        H_work.resize(N_in / 2);
        N = N_in / 2;
    }

    // determine decoding order (descending theta)
    std::vector<int> order(U);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return theta(a) > theta(b); });

    VectorXd stheta(U);
    for (int i = 0; i < U; ++i) stheta(i) = theta(order[i]);

    // delta = stheta - [stheta(2:end); 0]
    VectorXd delta(U);
    for (int i = 0; i < U - 1; ++i) {
        delta(i) = stheta(i) - stheta(i + 1);
    }
    delta(U - 1) = stheta(U - 1);

    // build sorted index mapping
    std::vector<int> idx_end_orig(U), idx_start_orig(U);
    idx_start_orig[0] = 0;
    idx_end_orig[0] = Lxu[0] - 1;
    for (int u = 1; u < U; ++u) {
        idx_start_orig[u] = idx_end_orig[u - 1] + 1;
        idx_end_orig[u] = idx_start_orig[u] + Lxu[u] - 1;
    }

    std::vector<int> Lxu_sorted(U);
    std::vector<int> idx_start_sorted(U), idx_end_sorted(U);
    for (int i = 0; i < U; ++i) {
        Lxu_sorted[i] = Lxu[order[i]];
        idx_start_sorted[i] = idx_start_orig[order[i]];
        idx_end_sorted[i] = idx_end_orig[order[i]];
    }

    int num_threads = config.num_threads;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    // bisection on lambda
    double lambda_min = 1e-14;
    double lambda_max = 1e14;
    const double tol = config.bisect_tol;
    const int max_iter = config.max_iter;

    MatrixXd best_Eun;
    std::vector<std::vector<MatrixXcd>> best_Rxx;
    double best_lambda = 1.0;

    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        double lambda = std::sqrt(lambda_min * lambda_max);

        MatrixXd Eun;
        std::vector<std::vector<MatrixXcd>> Rxx_sorted;

        eval_lagfunc(H_work, Lxu_sorted, idx_start_sorted, idx_end_sorted,
                     delta, lambda, Eun, Rxx_sorted);

        double E_current = Eun.sum();

        best_Eun = Eun;
        best_Rxx = Rxx_sorted;
        best_lambda = lambda;

        if (std::abs(E_current - Etotal) / std::max(Etotal, 1e-12) < tol) {
            break;
        }

        if (E_current > Etotal) {
            lambda_min = lambda;
        } else {
            lambda_max = lambda;
        }
    }

    // scale covariances so total energy equals exactly Etotal
    double E_total_used = best_Eun.sum();
    if (E_total_used > 1e-12) {
        double scale = Etotal / E_total_used;
        for (int u = 0; u < U; ++u) {
            for (int n = 0; n < N; ++n) {
                best_Rxx[u][n] *= scale;
                best_Eun(u, n) *= scale;
            }
        }
    }

    // compute rates from final covariances (in sorted order)
    MatrixXd bun_sorted;
    compute_rates(H_work, best_Rxx, Lxu_sorted, idx_start_sorted,
                  idx_end_sorted, cb, bun_sorted);

    // map results back to original user ordering
    std::vector<int> inv_order(U);
    for (int i = 0; i < U; ++i) inv_order[order[i]] = i;

    MatrixXd Eun_out(U, N);
    MatrixXd bun_out(U, N);
    std::vector<std::vector<MatrixXcd>> Rxx_out(U, std::vector<MatrixXcd>(N));

    for (int u = 0; u < U; ++u) {
        int su = inv_order[u];
        for (int n = 0; n < N; ++n) {
            Eun_out(u, n) = best_Eun(su, n);
            bun_out(u, n) = bun_sorted(su, n);
            Rxx_out[u][n] = best_Rxx[su][n];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double>(end_time - start_time).count();

    MaxRESMACResult result;
    result.Rxx = std::move(Rxx_out);
    result.Eun = Eun_out;
    result.w = VectorXd::Constant(U, best_lambda);
    result.bun = bun_out;
    result.weighted_rate = (theta.transpose() * bun_out.rowwise().sum())(0);
    result.iterations = iter + 1;
    result.elapsed_sec = elapsed;

    return result;
}

}  // namespace maxresmac
