/**
 * @file linalg_utils.hpp
 * @brief Linear algebra utilities for maxRESMAC MIMO optimization
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace maxresmac {

using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using MatrixXd = Eigen::MatrixXd;
using VectorXcd = Eigen::VectorXcd;
using VectorXd = Eigen::VectorXd;

inline MatrixXcd make_hermitian(const MatrixXcd& M) {
    return 0.5 * (M + M.adjoint());
}

/**
 * @brief Numerically stable log-determinant via Cholesky
 *
 * Mirrors the MATLAB logdet_stable function in maxRESMACMIMO.m
 */
inline double logdet_spd(const MatrixXcd& M) {
    const int n = M.rows();
    if (n == 0) return 0.0;

    MatrixXcd H = make_hermitian(M);

    Eigen::LLT<MatrixXcd> llt(H);
    if (llt.info() == Eigen::Success) {
        double val = 0.0;
        for (int i = 0; i < n; ++i) {
            val += 2.0 * std::log(std::real(llt.matrixL()(i, i)));
        }
        return val;
    }

    H += 1e-12 * MatrixXcd::Identity(n, n);
    Eigen::LLT<MatrixXcd> llt2(H);
    if (llt2.info() == Eigen::Success) {
        double val = 0.0;
        for (int i = 0; i < n; ++i) {
            val += 2.0 * std::log(std::real(llt2.matrixL()(i, i)));
        }
        return val;
    }

    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H);
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
        double ev = std::max(es.eigenvalues()(i), 1e-15);
        val += std::log(ev);
    }
    return val;
}

/**
 * @brief Cholesky factorization with inverse and logdet
 *
 * Mirrors the forward pass in eval_gradobj.m:
 *   [L, flag] = chol(S, 'lower');
 *   logdet = 2*sum(log(diag(L)));
 *   invS = (L\I)' * (L\I);
 */
inline bool cholesky_with_inv(const MatrixXcd& M, MatrixXcd& Minv,
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

    const auto& L = llt.matrixL();
    logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += 2.0 * std::log(std::real(L(i, i)));
    }

    Minv = llt.solve(MatrixXcd::Identity(n, n));
    return true;
}

inline void pack_complex(const MatrixXcd& M, VectorXd& vec, int offset = 0) {
    const int n = M.size();
    for (int i = 0; i < n; ++i) {
        vec(offset + i) = std::real(M.data()[i]);
        vec(offset + n + i) = std::imag(M.data()[i]);
    }
}

inline void unpack_complex(const VectorXd& vec, int offset, int rows, int cols,
                           MatrixXcd& M) {
    const int n = rows * cols;
    M.resize(rows, cols);
    for (int i = 0; i < n; ++i) {
        M.data()[i] = Complex(vec(offset + i), vec(offset + n + i));
    }
}

}  // namespace maxresmac
