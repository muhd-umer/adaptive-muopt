/**
 * @file linalg_utils.hpp
 * @brief Linear algebra utilities for minPMAC MIMO optimization
 *
 * Provides numerically stable log-determinant, Cholesky factorization,
 * water-filling, and matrix helper functions.
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <vector>

namespace minpmac {

using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using MatrixXd = Eigen::MatrixXd;
using VectorXcd = Eigen::VectorXcd;
using VectorXd = Eigen::VectorXd;

/**
 * @brief Numerically stable log-determinant for SPD matrices
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

/**
 * @brief Water-filling for single-user power allocation
 *
 * Given channel gains gn and target rate b_bar per tone,
 * computes optimal energy allocation.
 *
 * @param gn Channel gains per tone
 * @param b_bar Target rate per tone (bits)
 * @param cb Baseband type (1=complex, 2=real)
 * @param en Output energy allocation
 * @param bn Output bit allocation
 */
inline void waterfill(const VectorXd& gn, double b_bar, int cb, VectorXd& en,
                      VectorXd& bn) {
    const int N = gn.size();
    en.setZero(N);
    bn.setZero(N);

    if (N == 0) return;

    std::vector<std::pair<double, int>> sorted(N);
    for (int i = 0; i < N; ++i) {
        sorted[i] = {gn(i), i};
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int Nstar = N;
    for (int i = N - 1; i >= 0; --i) {
        if (sorted[i].first > 1e-15) break;
        --Nstar;
    }

    if (Nstar == 0) return;

    const double gap = 1.0;

    while (Nstar > 0) {
        double logK = std::log(gap);
        for (int i = 0; i < Nstar; ++i) {
            logK -= std::log(sorted[i].first) / Nstar;
        }
        logK += N * b_bar * 2.0 * std::log(2.0) / Nstar;
        double K = std::exp(logK);

        double En_min = K - gap / sorted[Nstar - 1].first;
        if (En_min >= 0) {
            for (int i = 0; i < Nstar; ++i) {
                int idx = sorted[i].second;
                double Ei = K - gap / sorted[i].first;
                en(idx) = Ei;
                bn(idx) = std::log2(K * sorted[i].first / gap) / cb;
            }
            break;
        }
        --Nstar;
    }
}

/**
 * @brief Compute maximum singular value of a matrix
 */
inline double max_singular_value(const MatrixXcd& M) {
    if (M.rows() == 0 || M.cols() == 0) return 0.0;
    Eigen::JacobiSVD<MatrixXcd> svd(M);
    return svd.singularValues()(0);
}

/**
 * @brief Compute matrix square root via eigendecomposition
 */
inline MatrixXcd matrix_sqrt(const MatrixXcd& M) {
    const int n = M.rows();
    MatrixXcd H = make_hermitian(M);

    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H);
    if (es.info() != Eigen::Success) {
        return MatrixXcd::Zero(n, n);
    }

    VectorXd d = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().adjoint();
}

/**
 * @brief Solve linear system for SPD matrix via Cholesky
 */
inline MatrixXcd spd_solve(const MatrixXcd& A, const MatrixXcd& B) {
    const int n = A.rows();
    MatrixXcd H = make_hermitian(A);

    Eigen::LLT<MatrixXcd> llt(H);
    if (llt.info() != Eigen::Success) {
        H += 1e-9 * MatrixXcd::Identity(n, n);
        llt.compute(H);
    }

    return llt.solve(B);
}

}  // namespace minpmac
