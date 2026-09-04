/**
 * @file linalg_utils.hpp
 * @brief Linear algebra utilities for maxRMAC MIMO optimization
 *
 * Provides numerically stable log-determinant, Cholesky factorization,
 * and matrix helper functions optimized for small matrices.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>

namespace maxrmac {

using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using MatrixXd = Eigen::MatrixXd;
using VectorXcd = Eigen::VectorXcd;
using VectorXd = Eigen::VectorXd;

/**
 * @brief Numerically stable log-determinant for SPD matrices
 *
 * Uses Cholesky decomposition with automatic regularization if needed.
 *
 * @param M Symmetric positive definite matrix
 * @return log|M|, or -inf if matrix is singular
 */
inline double logdet_spd(const MatrixXcd& M) {
    const int n = M.rows();
    if (n == 0) return 0.0;

    MatrixXcd H = 0.5 * (M + M.adjoint());

    Eigen::LLT<MatrixXcd> llt(H);
    if (llt.info() == Eigen::Success) {
        double logdet = 0.0;
        for (int i = 0; i < n; ++i) {
            logdet += 2.0 * std::log(std::real(llt.matrixL()(i, i)));
        }
        return logdet;
    }

    H += 1e-9 * MatrixXcd::Identity(n, n);
    Eigen::LLT<MatrixXcd> llt2(H);
    if (llt2.info() == Eigen::Success) {
        double logdet = 0.0;
        for (int i = 0; i < n; ++i) {
            logdet += 2.0 * std::log(std::real(llt2.matrixL()(i, i)));
        }
        return logdet;
    }

    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H);
    if (es.info() != Eigen::Success) {
        return -std::numeric_limits<double>::infinity();
    }

    double logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        double ev = std::max(es.eigenvalues()(i), 1e-15);
        logdet += std::log(ev);
    }
    return logdet;
}

/**
 * @brief Compute Cholesky factor and inverse for SPD matrix
 *
 * @param M Input SPD matrix (modified in-place as workspace)
 * @param L Output lower Cholesky factor
 * @param Minv Output inverse of M (computed via L)
 * @param logdet Output log-determinant
 * @return true if successful, false if singular
 */
inline bool cholesky_with_inv(const MatrixXcd& M, MatrixXcd& L, MatrixXcd& Minv,
                              double& logdet) {
    const int n = M.rows();
    if (n == 0) {
        logdet = 0.0;
        return true;
    }

    Eigen::LLT<MatrixXcd> llt(M);
    if (llt.info() != Eigen::Success) {
        MatrixXcd H = M + 1e-9 * MatrixXcd::Identity(n, n);
        llt.compute(H);
        if (llt.info() != Eigen::Success) {
            return false;
        }
    }

    L = llt.matrixL();

    logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += 2.0 * std::log(std::real(L(i, i)));
    }

    Minv = llt.solve(MatrixXcd::Identity(n, n));

    return true;
}

/**
 * @brief Compute matrix inverse for SPD matrix via Cholesky
 *
 * @param M Input SPD matrix
 * @param Minv Output inverse matrix
 * @return true if successful
 */
inline bool spd_inverse(const MatrixXcd& M, MatrixXcd& Minv) {
    const int n = M.rows();
    MatrixXcd H = 0.5 * (M + M.adjoint());

    Eigen::LLT<MatrixXcd> llt(H);
    if (llt.info() != Eigen::Success) {
        H += 1e-9 * MatrixXcd::Identity(n, n);
        llt.compute(H);
        if (llt.info() != Eigen::Success) {
            return false;
        }
    }

    Minv = llt.solve(MatrixXcd::Identity(n, n));
    return true;
}

/**
 * @brief Make matrix Hermitian by averaging with its adjoint
 */
inline MatrixXcd make_hermitian(const MatrixXcd& M) {
    return 0.5 * (M + M.adjoint());
}

/**
 * @brief Check if matrix is positive semi-definite
 */
inline bool is_psd(const MatrixXcd& M, double tol = -1e-9) {
    MatrixXcd H = make_hermitian(M);
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) return false;
    return es.eigenvalues().minCoeff() >= tol;
}

/**
 * @brief Compute Frobenius norm squared for complex matrix
 */
inline double frobenius_norm_sq(const MatrixXcd& M) { return M.squaredNorm(); }

/**
 * @brief Pack complex matrix into real vector (interleaved real/imag)
 */
inline void pack_complex(const MatrixXcd& M, VectorXd& vec, int offset = 0) {
    const int n = M.size();
    for (int i = 0; i < n; ++i) {
        vec(offset + i) = std::real(M.data()[i]);
        vec(offset + n + i) = std::imag(M.data()[i]);
    }
}

/**
 * @brief Unpack real vector into complex matrix
 */
inline void unpack_complex(const VectorXd& vec, int offset, int rows, int cols,
                           MatrixXcd& M) {
    const int n = rows * cols;
    M.resize(rows, cols);
    for (int i = 0; i < n; ++i) {
        M.data()[i] = Complex(vec(offset + i), vec(offset + n + i));
    }
}

}  // namespace maxrmac
