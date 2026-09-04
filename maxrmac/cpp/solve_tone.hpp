/**
 * @file solve_tone.hpp
 * @brief Per-tone L-BFGS optimization for maxRMAC MIMO
 *
 * Solves the per-tone Lagrangian optimization problem using L-BFGS
 * over Cholesky factors of covariance matrices.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <vector>

#include "linalg_utils.hpp"

namespace maxrmac {

/**
 * @brief Result from per-tone optimization
 */
struct ToneResult {
    double f;

    VectorXd b;

    std::vector<MatrixXcd> Rxx;

    VectorXd E;
};

/**
 * @brief L-BFGS state for warm-starting
 */
struct LBFGSState {
    std::vector<MatrixXcd> B_blocks;

    std::vector<VectorXd> S_hist;

    std::vector<VectorXd> Y_hist;

    std::vector<int> order;

    std::vector<int> block_sizes;

    bool valid = false;
};

/**
 * @brief Workspace for per-tone optimization (avoids allocations)
 */
struct ToneWorkspace {
    std::vector<MatrixXcd> H_blocks;

    VectorXd x, x_new, x_trial;
    VectorXd grad, grad_new, grad_trial;
    VectorXd dir;

    std::vector<MatrixXcd> B_blocks;
    std::vector<MatrixXcd> R_blocks;
    std::vector<MatrixXcd> invS;
    VectorXd logdet_S;

    MatrixXcd S, S_prev, HRH;
    MatrixXcd L, Linv;
    MatrixXcd tail, grad_R, grad_B;

    VectorXd alpha, rho, q, r;

    void resize(int U, int Ly, int max_Lxu, int total_params) {
        H_blocks.resize(U);
        B_blocks.resize(U);
        R_blocks.resize(U);
        invS.resize(U);
        logdet_S.resize(U);

        x.resize(total_params);
        x_new.resize(total_params);
        x_trial.resize(total_params);
        grad.resize(total_params);
        grad_new.resize(total_params);
        grad_trial.resize(total_params);
        dir.resize(total_params);

        S.resize(Ly, Ly);
        S_prev.resize(Ly, Ly);
        HRH.resize(Ly, Ly);
        L.resize(Ly, Ly);
        Linv.resize(Ly, Ly);
        tail.resize(Ly, Ly);

        for (int u = 0; u < U; ++u) {
            invS[u].resize(Ly, Ly);
        }
    }
};

/**
 * @brief Solve per-tone Lagrangian optimization
 *
 * Maximizes: sum_u delta(u) * log|S_u| - sum_u w(u) * trace(Rxx_u)
 * where S_u = I + sum_{j=1}^u H_j * Rxx_j * H_j'
 *
 * @param Hn Channel matrix for this tone [Ly x Ltot]
 * @param theta Per-user rate weights [U]
 * @param w Per-user energy weights (Lagrange multipliers) [U]
 * @param Lxu Per-user antenna counts [U]
 * @param idx_start Starting column index for each user
 * @param idx_end Ending column index for each user
 * @param state_in Optional warm-start state
 * @param workspace Pre-allocated workspace
 * @return ToneResult with optimized covariances and rates
 */
ToneResult solve_tone(const MatrixXcd& Hn, const VectorXd& theta,
                      const VectorXd& w, const std::vector<int>& Lxu,
                      const std::vector<int>& idx_start,
                      const std::vector<int>& idx_end, LBFGSState& state_in,
                      ToneWorkspace& workspace);

/**
 * @brief L-BFGS configuration parameters
 */
struct LBFGSConfig {
    int max_iter = 200;
    double tol_grad = 1e-6;
    int m_hist = 12;
    double step_init = 1.0;
    double step_min = 1e-12;
    double beta = 0.5;
    double armijo_c = 1e-4;
    int max_linesearch = 24;
};

}  // namespace maxrmac
