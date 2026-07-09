/**
 * @file maxresmac.hpp
 * @brief Multi-user MIMO MAC rate maximization with total energy constraint
 *
 * Maximizes weighted sum-rate subject to a single total sum-energy
 * constraint using bisection on Lagrange multiplier lambda.
 * Self-contained: does not depend on maxrmac.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <string>
#include <vector>

#include "linalg_utils.hpp"

namespace maxresmac {

struct MaxRESMACResult {
    std::vector<std::vector<MatrixXcd>> Rxx;

    MatrixXd Eun;

    VectorXd w;

    MatrixXd bun;

    double weighted_rate;

    int iterations;

    double elapsed_sec;
};

struct MaxRESMACConfig {
    double bisect_tol = 1e-8;

    int max_iter = 120;

    int num_threads = 0;

    bool verbose = false;
};

/**
 * @brief Maximize weighted sum rate for MIMO MAC with total energy constraint
 *
 * Solves: max sum_u theta(u) * R_u
 *         s.t. sum_u sum_n trace(Rxx_u,n) <= Etotal
 *              Rxx_u,n >= 0  (PSD)
 *
 * @param H Channel tensor as vector of N matrices [Ly x sum(Lxu)]
 * @param Lxu Antennas per user (length U)
 * @param Etotal Total sum-energy constraint (scalar)
 * @param theta Per-user rate weights (length U)
 * @param cb Baseband type: 1 for complex, 2 for real
 * @param config Algorithm configuration
 */
MaxRESMACResult maxRESMACMIMO(
    const std::vector<MatrixXcd>& H, const std::vector<int>& Lxu, double Etotal,
    const VectorXd& theta, int cb,
    const MaxRESMACConfig& config = MaxRESMACConfig());

}  // namespace maxresmac
