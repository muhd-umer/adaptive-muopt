/**
 * @file solve_tone.hpp
 * @brief Per-tone L-BFGS optimization for maxRESMAC MIMO
 *
 * Direct C++ port of maxresmac/utils/solve_tone.m and eval_gradobj.m.
 * Solves: max sum_u delta(u)*log|S_u| - lambda*sum_u trace(Rxx_u)
 * where S_u = I + sum_{j=1}^u H_j*Rxx_j*H_j'
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <vector>

#include "linalg_utils.hpp"

namespace maxresmac {

struct ToneResult {
    std::vector<MatrixXcd> Rxx;
    VectorXd E;
};

struct LBFGSConfig {
    int max_iter = 200;
    double tol_grad = 1e-7;
    int m_hist = 10;
    double step_init = 1.0;
    double step_min = 1e-14;
    double beta = 0.5;
    double armijo_c = 1e-4;
    int max_linesearch = 30;
};

/**
 * @brief Solve per-tone Lagrangian optimization
 *
 * @param Hn Channel matrix for this tone [Ly x Ltot]
 * @param Lxu Per-user antenna counts (sorted order) [U]
 * @param idx_start Starting column index for each user (sorted, 0-based)
 * @param idx_end Ending column index for each user (sorted, 0-based inclusive)
 * @param delta Rate weight differences (sorted) [U]
 * @param lambda Scalar Lagrange multiplier for total energy
 * @return ToneResult with optimal covariances and energies
 */
ToneResult solve_tone(const MatrixXcd& Hn, const std::vector<int>& Lxu,
                      const std::vector<int>& idx_start,
                      const std::vector<int>& idx_end, const VectorXd& delta,
                      double lambda, const LBFGSConfig& cfg = LBFGSConfig());

}  // namespace maxresmac
