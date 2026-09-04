/**
 * @file minpmac.cpp
 * @brief Main minPMAC MIMO solver implementation
 *
 * Uses ellipsoid method for dual optimization with OpenMP-parallelized
 * per-tone L-BFGS optimization.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "minpmac.hpp"

#include <glpk.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

#include "solve_tone.hpp"

namespace minpmac {

namespace {

/**
 * @brief Initialize ellipsoid for dual optimization
 *
 * Uses water-filling approximation to estimate initial theta values.
 */
void init_ellipsoid(const std::vector<MatrixXcd>& H, const VectorXd& bu,
                    const VectorXd& w, int cb, const std::vector<int>& Lxu,
                    const std::vector<int>& idx_start,
                    const std::vector<int>& idx_end, MatrixXd& A,
                    VectorXd& theta_init) {
    const int N = H.size();
    const int U = Lxu.size();
    const int Ly = H[0].rows();

    VectorXd tmax(U);

    for (int u = 0; u < U; ++u) {
        MatrixXd en = MatrixXd::Zero(U, N);

        VectorXd b = bu;
        b(u) += 1.0;

        for (int u1 = U - 1; u1 >= 0; --u1) {
            std::vector<MatrixXcd> Rnoise(N);
            for (int n = 0; n < N; ++n) {
                Rnoise[n] = MatrixXcd::Identity(Ly, Ly);

                for (int u2 = U - 1; u2 >= 0; --u2) {
                    if (u2 != u1) {
                        MatrixXcd H_u2 = H[n].middleCols(
                            idx_start[u2], idx_end[u2] - idx_start[u2] + 1);

                        if (Lxu[u2] == 1) {
                            Rnoise[n] += en(u2, n) * H_u2 * H_u2.adjoint();
                        } else {
                            double energy_per_ant = en(u2, n) / Lxu[u2];
                            Rnoise[n] += energy_per_ant * H_u2 * H_u2.adjoint();
                        }
                    }
                }
            }

            VectorXd g_tones(N);
            for (int n = 0; n < N; ++n) {
                MatrixXcd H_u1 = H[n].middleCols(
                    idx_start[u1], idx_end[u1] - idx_start[u1] + 1);
                MatrixXcd Rsqrt_inv = spd_solve(matrix_sqrt(Rnoise[n]),
                                                MatrixXcd::Identity(Ly, Ly));
                MatrixXcd h_temp = Rsqrt_inv * H_u1;

                if (Lxu[u1] == 1) {
                    g_tones(n) = std::real((h_temp.adjoint() * h_temp)(0, 0));
                } else {
                    g_tones(n) = std::pow(max_singular_value(h_temp), 2);
                }
            }

            VectorXd bn_wf, en_wf;
            double b_bar = b(u1) / N;
            waterfill(g_tones, b_bar, cb, en_wf, bn_wf);

            for (int n = 0; n < N; ++n) {
                en(u1, n) = en_wf(n);
            }
        }

        double total_weighted_energy = 0.0;
        for (int v = 0; v < U; ++v) {
            total_weighted_energy += w(v) * en.row(v).sum();
        }
        tmax(u) = total_weighted_energy;
    }

    theta_init = tmax / 2.0;
    for (int u = 0; u < U; ++u) {
        theta_init(u) = std::max(theta_init(u), 0.1);
    }
    double radius_sq = theta_init.squaredNorm();
    A = std::max(radius_sq, 1.0) * MatrixXd::Identity(U, U);
}

/**
 * @brief Evaluate Lagrangian across all tones (parallelized)
 */
void eval_lagfunc(const std::vector<MatrixXcd>& H, const VectorXd& theta,
                  const VectorXd& w, const VectorXd& bu_scaled,
                  const std::vector<int>& Lxu,
                  const std::vector<int>& idx_start,
                  const std::vector<int>& idx_end,
                  std::vector<LBFGSState>& states,
                  std::vector<ToneWorkspace>& workspaces, double& f,
                  MatrixXd& bun, std::vector<MatrixXcd>& Rxx_all) {
    const int N = H.size();
    const int U = Lxu.size();
    const int Ltot = H[0].cols();

    bun.resize(U, N);
    Rxx_all.resize(N);

    std::vector<ToneResult> results(N);

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        results[n] = solve_tone(H[n], theta, w, Lxu, idx_start, idx_end,
                                states[n], workspaces[n]);
    }

    f = 0.0;
    for (int n = 0; n < N; ++n) {
        f += results[n].f;

        Rxx_all[n] = MatrixXcd::Zero(Ltot, Ltot);
        for (int u = 0; u < U; ++u) {
            bun(u, n) = results[n].b(u);

            int start = idx_start[u];
            int cols = idx_end[u] - idx_start[u] + 1;
            Rxx_all[n].block(start, start, cols, cols) = results[n].Rxx[u];
        }
    }

    f -= theta.dot(bu_scaled);
}

/**
 * @brief Cluster users by theta values
 */
void cluster_theta_values(const VectorXd& theta, double tol,
                          std::vector<std::vector<int>>& clusters,
                          VectorXd& theta_unique) {
    const int U = theta.size();

    std::vector<double> sorted_vals;
    for (int u = 0; u < U; ++u) {
        sorted_vals.push_back(theta(u));
    }
    std::sort(sorted_vals.begin(), sorted_vals.end());

    std::vector<double> unique_vals;
    for (double v : sorted_vals) {
        if (unique_vals.empty() || std::abs(v - unique_vals.back()) > tol) {
            unique_vals.push_back(v);
        }
    }

    theta_unique.resize(unique_vals.size());
    for (size_t i = 0; i < unique_vals.size(); ++i) {
        theta_unique(i) = unique_vals[i];
    }

    clusters.resize(unique_vals.size());
    for (int u = 0; u < U; ++u) {
        for (size_t c = 0; c < unique_vals.size(); ++c) {
            if (std::abs(theta(u) - unique_vals[c]) <= tol) {
                clusters[c].push_back(u);
                break;
            }
        }
    }
}

/**
 * @brief Generate all permutations of a vector
 */
void generate_permutations(const std::vector<int>& arr,
                           std::vector<std::vector<int>>& perms) {
    std::vector<int> sorted = arr;
    std::sort(sorted.begin(), sorted.end());
    perms.clear();
    do {
        perms.push_back(sorted);
    } while (std::next_permutation(sorted.begin(), sorted.end()));
}

/**
 * @brief Generate all decoding orders from clustered users
 */
void generate_decoding_orders(const std::vector<std::vector<int>>& clusters,
                              std::vector<std::vector<int>>& orders) {
    if (clusters.empty()) {
        orders.clear();
        return;
    }

    std::vector<std::vector<std::vector<int>>> cluster_perms(clusters.size());
    for (size_t c = 0; c < clusters.size(); ++c) {
        generate_permutations(clusters[c], cluster_perms[c]);
    }

    orders = cluster_perms[0];
    for (size_t c = 1; c < clusters.size(); ++c) {
        std::vector<std::vector<int>> new_orders;
        for (const auto& prev : orders) {
            for (const auto& curr : cluster_perms[c]) {
                std::vector<int> combined = prev;
                combined.insert(combined.end(), curr.begin(), curr.end());
                new_orders.push_back(combined);
            }
        }
        orders = std::move(new_orders);
    }
}

/**
 * @brief Compute rates for a given decoding order
 */
void compute_rates(const std::vector<MatrixXcd>& H,
                   const std::vector<MatrixXcd>& Rxx,
                   const std::vector<int>& order, int cb,
                   const std::vector<int>& Lxu,
                   const std::vector<int>& idx_start,
                   const std::vector<int>& idx_end, MatrixXd& bun,
                   VectorXd& bu) {
    const int N = H.size();
    const int U = Lxu.size();
    const int Ly = H[0].rows();

    bun.resize(U, N);
    bun.setZero();

    const double inv_log2_cb = 1.0 / (std::log(2.0) * cb);

    for (int n = 0; n < N; ++n) {
        const auto& Rn = Rxx[n];
        MatrixXcd eye_Ly = MatrixXcd::Identity(Ly, Ly);

        std::vector<MatrixXcd> suffix_cache(U);
        for (int k = 0; k < U; ++k) {
            int user_idx = order[k];
            int start = idx_start[user_idx];
            int cols = idx_end[user_idx] - idx_start[user_idx] + 1;
            MatrixXcd Hv = H[n].middleCols(start, cols);
            MatrixXcd Rv = Rn.block(start, start, cols, cols);
            MatrixXcd HRH = Hv * Rv * Hv.adjoint();
            suffix_cache[k] = make_hermitian(HRH);
        }

        MatrixXcd suffix_sum = MatrixXcd::Zero(Ly, Ly);
        double suffix_next_logdet = 0.0;

        for (int pos = U - 1; pos >= 0; --pos) {
            double logdet_interf = suffix_next_logdet;
            suffix_sum += suffix_cache[pos];
            MatrixXcd S = eye_Ly + suffix_sum;
            S = make_hermitian(S);

            double logdet_signal = logdet_spd(S);
            suffix_next_logdet = logdet_signal;

            int user_idx = order[pos];
            double rate_bits = (logdet_signal - logdet_interf) * inv_log2_cb;
            bun(user_idx, n) = std::real(rate_bits);
        }
    }

    bu = bun.rowwise().sum();
}

/**
 * @brief Solve time-sharing LP using GLPK (GNU Linear Programming Kit)
 *
 * Solves the LP problem (equivalent to MATLAB's linprog):
 *   minimize: 0 (feasibility)
 *   subject to: bu_vertices * w >= bu_min - tol
 *               sum(w) == 1
 *               w >= 0
 *
 * Uses GLPK's simplex solver for robust, accurate LP solving.
 */
bool solve_time_sharing(const MatrixXd& bu_vertices, const VectorXd& bu_min,
                        double tol, VectorXd& weights,
                        std::vector<bool>& active_idx) {
    const int U = bu_vertices.rows();

    const int K = bu_vertices.cols();

    weights.resize(K);
    active_idx.resize(K, false);

    weights.setConstant(1.0 / K);
    VectorXd bu_achieved = bu_vertices * weights;
    bool uniform_feasible = true;
    for (int u = 0; u < U; ++u) {
        if (bu_achieved(u) < bu_min(u) - tol) {
            uniform_feasible = false;
            break;
        }
    }
    if (uniform_feasible) {
        const double thresh = 1e-6;
        for (int k = 0; k < K; ++k) {
            active_idx[k] = (weights(k) > thresh);
        }
        return true;
    }

    glp_prob* lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);

    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF;

    glp_add_cols(lp, K);
    for (int k = 0; k < K; ++k) {
        glp_set_col_bnds(lp, k + 1, GLP_LO, 0.0, 0.0);

        glp_set_obj_coef(lp, k + 1, 0.0);
    }

    glp_add_rows(lp, U + 1);

    for (int u = 0; u < U; ++u) {
        glp_set_row_bnds(lp, u + 1, GLP_LO, bu_min(u) - tol, 0.0);
    }

    glp_set_row_bnds(lp, U + 1, GLP_FX, 1.0, 1.0);

    int num_elements = U * K + K;

    std::vector<int> ia(num_elements + 1);
    std::vector<int> ja(num_elements + 1);
    std::vector<double> ar(num_elements + 1);

    int idx = 1;

    for (int u = 0; u < U; ++u) {
        for (int k = 0; k < K; ++k) {
            ia[idx] = u + 1;
            ja[idx] = k + 1;
            ar[idx] = bu_vertices(u, k);
            ++idx;
        }
    }

    for (int k = 0; k < K; ++k) {
        ia[idx] = U + 1;
        ja[idx] = k + 1;
        ar[idx] = 1.0;
        ++idx;
    }

    glp_load_matrix(lp, num_elements, ia.data(), ja.data(), ar.data());

    int ret = glp_simplex(lp, &parm);

    bool feasible = false;
    if (ret == 0) {
        int status = glp_get_status(lp);
        if (status == GLP_OPT || status == GLP_FEAS) {
            feasible = true;

            for (int k = 0; k < K; ++k) {
                weights(k) = glp_get_col_prim(lp, k + 1);
            }
        }
    }

    glp_delete_prob(lp);

    if (!feasible) {
        weights.setZero();
        return false;
    }

    for (int k = 0; k < K; ++k) {
        if (weights(k) < 0) weights(k) = 0;
    }

    double sum = weights.sum();
    if (sum < 1e-10) {
        weights.setZero();
        return false;
    }
    weights /= sum;

    bu_achieved = bu_vertices * weights;
    for (int u = 0; u < U; ++u) {
        if (bu_achieved(u) < bu_min(u) - tol - 1e-6) {
            weights.setZero();
            return false;
        }
    }

    const double thresh = 1e-6;
    for (int k = 0; k < K; ++k) {
        if (weights(k) < thresh) {
            weights(k) = 0.0;
        } else {
            active_idx[k] = true;
        }
    }

    sum = weights.sum();
    if (sum > 0) {
        weights /= sum;
    } else {
        return false;
    }

    return true;
}

/**
 * @brief Compute average energies per user
 */
VectorXd compute_avg_energies(const std::vector<MatrixXcd>& Rxx,
                              const std::vector<int>& Lxu,
                              const std::vector<int>& idx_start,
                              const std::vector<int>& idx_end) {
    const int N = Rxx.size();
    const int U = Lxu.size();

    VectorXd Eu_avg(U);
    Eu_avg.setZero();

    for (int u = 0; u < U; ++u) {
        int start = idx_start[u];
        int cols = idx_end[u] - idx_start[u] + 1;
        double total = 0.0;
        for (int n = 0; n < N; ++n) {
            total += std::real(Rxx[n].block(start, start, cols, cols).trace());
        }
        Eu_avg(u) = total / N;
    }

    return Eu_avg;
}

}  // namespace

MinPMACResult minPMACMIMO(const std::vector<MatrixXcd>& H,
                          const std::vector<int>& Lxu, const VectorXd& bu_min,
                          const VectorXd& w, int cb,
                          const MinPMACConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int Ly = H[0].rows();
    const int Ltot = H[0].cols();
    const int N = H.size();
    const int U = Lxu.size();

    if (bu_min.size() != U) {
        throw std::invalid_argument("bu_min must have length U");
    }
    if (w.size() != U) {
        throw std::invalid_argument("w must have length U");
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

    VectorXd bu_internal = bu_min / (3.0 - cb);
    VectorXd bu_scaled = bu_internal * std::log(2.0);

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
    VectorXd theta;
    init_ellipsoid(H, bu_internal, w, cb, Lxu, idx_start, idx_end, A, theta);

    if (config.verbose) {
        std::cout << "minPMACMIMO: " << U << " users, " << N << " tones, " << Ly
                  << " RX antennas" << std::endl;
    }

    double f;
    MatrixXd bun;
    std::vector<MatrixXcd> Rxx_all;

    int iter = 0;
    const double inv_U_plus_1 = 1.0 / (U + 1);
    const double U_sq = static_cast<double>(U * U);
    const double scale_A = U_sq / (U_sq - 1);
    const double scale_A_gt = 2.0 / (U + 1);

    while (true) {
        ++iter;

        eval_lagfunc(H, theta, w, bu_scaled, Lxu, idx_start, idx_end, states,
                     workspaces, f, bun, Rxx_all);

        VectorXd g = bun.rowwise().sum() - bu_scaled;

        double gAg = g.transpose() * A * g;
        if (std::sqrt(gAg) <= config.err_tol) {
            break;
        }
        if (iter > config.max_iter) {
            if (config.verbose) {
                std::cout << "Warning: ellipsoid did not converge in " << iter
                          << " iterations" << std::endl;
            }
            break;
        }

        double inv_sqrt_gAg = 1.0 / std::sqrt(gAg);
        VectorXd gt = g * inv_sqrt_gAg;
        VectorXd Agt = A * gt;
        theta -= inv_U_plus_1 * Agt;
        A = scale_A * (A - scale_A_gt * Agt * gt.transpose() * A);

        for (int u = 0; u < U; ++u) {
            while (theta(u) < 0) {
                VectorXd g_proj = VectorXd::Zero(U);
                g_proj(u) = -1.0;
                double gAg_proj = A(u, u);
                double inv_sqrt_gAg_proj = 1.0 / std::sqrt(gAg_proj);
                VectorXd gt_proj = g_proj * inv_sqrt_gAg_proj;
                VectorXd Agt_proj = A * gt_proj;
                theta -= inv_U_plus_1 * Agt_proj;
                A = scale_A *
                    (A - scale_A_gt * Agt_proj * gt_proj.transpose() * A);
            }
        }
    }

    eval_lagfunc(H, theta, w, bu_scaled, Lxu, idx_start, idx_end, states,
                 workspaces, f, bun, Rxx_all);

    std::vector<std::vector<int>> clusters;
    VectorXd theta_unique;
    cluster_theta_values(theta, 1e-8, clusters, theta_unique);

    std::vector<std::vector<int>> all_orders;
    generate_decoding_orders(clusters, all_orders);

    if (config.verbose) {
        std::cout << "Found " << clusters.size() << " clusters, "
                  << all_orders.size() << " decoding orders" << std::endl;
    }

    MinPMACResult result;
    result.theta = theta;
    result.iterations = iter;
    result.Rxx = Rxx_all;

    if (all_orders.size() == 1) {
        const auto& order = all_orders[0];
        MatrixXd bun_order;
        VectorXd bu_order;
        compute_rates(H, Rxx_all, order, cb, Lxu, idx_start, idx_end, bun_order,
                      bu_order);

        result.feas_flag = 1;
        result.bu_a = bu_order;
        result.bun = bun_order;
        result.orderings.push_back(order);

    } else {
        const int num_orders = all_orders.size();
        MatrixXd bu_vertices(U, num_orders);
        std::vector<MatrixXd> bun_vertices(num_orders);

        for (int k = 0; k < num_orders; ++k) {
            MatrixXd bun_k;
            VectorXd bu_k;
            compute_rates(H, Rxx_all, all_orders[k], cb, Lxu, idx_start,
                          idx_end, bun_k, bu_k);
            bu_vertices.col(k) = bu_k;
            bun_vertices[k] = bun_k;
        }

        VectorXd weights;
        std::vector<bool> active_idx;
        double tol = 1e-3;
        if (!solve_time_sharing(bu_vertices, bu_min, tol, weights,
                                active_idx)) {
            result.feas_flag = 0;
            result.bu_a = VectorXd::Zero(U);
            result.bun = MatrixXd::Zero(U, N);
        } else {
            result.feas_flag = 2;

            VectorXd bu_avg = VectorXd::Zero(U);
            MatrixXd bun_avg = MatrixXd::Zero(U, N);
            VectorXd active_weights;

            int count = 0;
            for (int k = 0; k < num_orders; ++k) {
                if (active_idx[k] && weights(k) > 1e-8) {
                    ++count;
                }
            }
            active_weights.resize(count);

            int idx = 0;
            for (int k = 0; k < num_orders; ++k) {
                if (active_idx[k] && weights(k) > 1e-8) {
                    bu_avg += weights(k) * bu_vertices.col(k);
                    bun_avg += weights(k) * bun_vertices[k];
                    result.orderings.push_back(all_orders[k]);
                    active_weights(idx++) = weights(k);
                }
            }

            result.bu_a = bu_avg;
            result.bun = bun_avg;
            result.frac = active_weights;
        }
    }

    result.Eu_avg = compute_avg_energies(Rxx_all, Lxu, idx_start, idx_end);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_sec =
        std::chrono::duration<double>(end_time - start_time).count();

    if (config.verbose) {
        std::cout << "minPMACMIMO completed in " << result.elapsed_sec * 1000
                  << " ms, feas_flag=" << result.feas_flag << std::endl;
    }

    return result;
}

MinPMACResult minPMACMIMO(const std::vector<MatrixXcd>& H, int Lxu_scalar,
                          int U, const VectorXd& bu_min, const VectorXd& w,
                          int cb, const MinPMACConfig& config) {
    std::vector<int> Lxu(U, Lxu_scalar);
    return minPMACMIMO(H, Lxu, bu_min, w, cb, config);
}

}  // namespace minpmac
