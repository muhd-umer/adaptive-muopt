/**
 * @file maxrmac_mimo.hpp
 * @brief Multi-user MIMO MAC rate maximization solver
 *
 * Implements weighted sum-rate maximization subject to per-user
 * energy constraints using ellipsoid method + L-BFGS.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <string>
#include <vector>

#include "linalg_utils.hpp"

namespace maxrmac {

/**
 * @brief Result from maxRMACMIMO solver
 */
struct MaxRMACResult {
    std::vector<std::vector<MatrixXcd>> Rxx;

    MatrixXd Eun;

    VectorXd w;

    MatrixXd bun;

    double weighted_rate;

    int iterations;

    double elapsed_sec;
};

/**
 * @brief Algorithm configuration
 */
struct MaxRMACConfig {
    double err_tol = 1e-9;

    int max_iter = 2000;

    int num_threads = 0;

    bool verbose = false;
};

/**
 * @brief Maximize weighted sum rate for multi-user MIMO MAC
 *
 * Solves: max sum_u theta(u) * R_u
 *         s.t. sum_n trace(Rxx_u,n) <= Eu(u)  for all u
 *              Rxx_u,n >= 0  (PSD)
 *
 * @param H Channel tensor [Ly x sum(Lxu) x N] as vector of N matrices
 * @param Lxu Antennas per user (length U)
 * @param Eu Per-user energy constraints (length U)
 * @param theta Per-user rate weights (length U)
 * @param cb Baseband type: 1 for complex, 2 for real
 * @param config Algorithm configuration
 * @return MaxRMACResult with optimal covariances, rates, and diagnostics
 */
MaxRMACResult maxRMACMIMO(const std::vector<MatrixXcd>& H,
                          const std::vector<int>& Lxu, const VectorXd& Eu,
                          const VectorXd& theta, int cb,
                          const MaxRMACConfig& config = MaxRMACConfig());

/**
 * @brief Convenience overload for uniform antennas
 *
 * @param H Channel tensor as single 3D array [Ly x (U*Lxu) x N]
 * @param Lxu_scalar Antennas per user (same for all)
 * @param U Number of users
 * @param Eu Per-user energy constraints
 * @param theta Per-user rate weights
 * @param cb Baseband type
 * @param config Algorithm configuration
 */
MaxRMACResult maxRMACMIMO(const std::vector<MatrixXcd>& H, int Lxu_scalar,
                          int U, const VectorXd& Eu, const VectorXd& theta,
                          int cb,
                          const MaxRMACConfig& config = MaxRMACConfig());

}  // namespace maxrmac
