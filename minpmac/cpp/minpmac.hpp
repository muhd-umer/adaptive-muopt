/**
 * @file minpmac.hpp
 * @brief Multi-user MIMO MAC minimum power solver
 *
 * Implements minimum weighted energy optimization subject to per-user
 * rate constraints using ellipsoid method + L-BFGS.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <string>
#include <vector>

#include "linalg_utils.hpp"

namespace minpmac {

/**
 * @brief Result from minPMACMIMO solver
 */
struct MinPMACResult {
    int feas_flag;

    VectorXd bu_a;

    MatrixXd bun;

    std::vector<MatrixXcd> Rxx;

    VectorXd frac;

    VectorXd Eu_avg;

    VectorXd theta;

    std::vector<std::vector<int>> orderings;

    int iterations;

    double elapsed_sec;
};

/**
 * @brief Algorithm configuration
 */
struct MinPMACConfig {
    double err_tol = 1e-9;

    int max_iter = 2000;

    int num_threads = 0;

    bool verbose = false;
};

/**
 * @brief Minimize weighted energy for multi-user MIMO MAC
 *
 * Solves: min sum_u w(u) * E_u
 *         s.t. sum_n b_u,n >= bu_min(u)  for all u
 *              Rxx_u,n >= 0  (PSD)
 *
 * @param H Channel tensor [Ly x sum(Lxu) x N] as vector of N matrices
 * @param Lxu Antennas per user (length U)
 * @param bu_min Per-user rate targets in bits (length U)
 * @param w Per-user energy weights (length U)
 * @param cb Baseband type: 1 for complex, 2 for real
 * @param config Algorithm configuration
 * @return MinPMACResult with optimal covariances, rates, and diagnostics
 */
MinPMACResult minPMACMIMO(const std::vector<MatrixXcd>& H,
                          const std::vector<int>& Lxu, const VectorXd& bu_min,
                          const VectorXd& w, int cb,
                          const MinPMACConfig& config = MinPMACConfig());

/**
 * @brief Convenience overload for uniform antennas
 */
MinPMACResult minPMACMIMO(const std::vector<MatrixXcd>& H, int Lxu_scalar,
                          int U, const VectorXd& bu_min, const VectorXd& w,
                          int cb,
                          const MinPMACConfig& config = MinPMACConfig());

}  // namespace minpmac
