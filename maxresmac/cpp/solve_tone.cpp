/**
 * @file solve_tone.cpp
 * @brief Per-tone L-BFGS optimization -- direct port of
 * maxresmac/utils/solve_tone.m
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "solve_tone.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace maxresmac {

namespace {

void pack_blocks(const std::vector<MatrixXcd>& blocks,
                 const std::vector<int>& Lxu, VectorXd& vec) {
    int offset = 0;
    for (size_t u = 0; u < blocks.size(); ++u) {
        const auto& B = blocks[u];
        const int n = B.size();
        for (int i = 0; i < n; ++i) {
            vec(offset + i) = std::real(B.data()[i]);
            vec(offset + n + i) = std::imag(B.data()[i]);
        }
        offset += 2 * n;
    }
}

void unpack_blocks(const VectorXd& vec, const std::vector<int>& Lxu,
                   std::vector<MatrixXcd>& blocks) {
    int offset = 0;
    for (size_t u = 0; u < Lxu.size(); ++u) {
        const int Lu = Lxu[u];
        const int n = Lu * Lu;
        blocks[u].resize(Lu, Lu);
        for (int i = 0; i < n; ++i) {
            blocks[u].data()[i] = Complex(vec(offset + i), vec(offset + n + i));
        }
        offset += 2 * n;
    }
}

/**
 * @brief Evaluate objective and gradient -- direct port of eval_gradobj.m
 *
 * phi = -(sum delta.*logdet_S - lambda*energy)  (negated for minimization)
 * grad w.r.t. B_u = -2*(H'*tail*H - lambda*I)*B_u
 */
bool eval_gradobj(const std::vector<MatrixXcd>& B_blocks, const MatrixXcd& Hn,
                  const VectorXd& delta, double lambda,
                  const std::vector<int>& Lxu,
                  const std::vector<int>& idx_start,
                  const std::vector<int>& idx_end, int U, int Ly, double& phi,
                  std::vector<MatrixXcd>& grad_blocks,
                  std::vector<MatrixXcd>& R_blocks) {
    MatrixXcd S = MatrixXcd::Identity(Ly, Ly);
    VectorXd logdet_S(U);
    std::vector<MatrixXcd> invS(U);
    R_blocks.resize(U);

    // forward pass: compute S_u and log|S_u|
    for (int u = 0; u < U; ++u) {
        const auto& Bu = B_blocks[u];
        R_blocks[u] = Bu * Bu.adjoint();

        MatrixXcd Hu =
            Hn.middleCols(idx_start[u], idx_end[u] - idx_start[u] + 1);
        MatrixXcd HRH = Hu * R_blocks[u] * Hu.adjoint();
        S += HRH;
        S = make_hermitian(S);

        double ld;
        if (!cholesky_with_inv(S, invS[u], ld)) {
            phi = std::numeric_limits<double>::max();
            return false;
        }
        logdet_S(u) = ld;
    }

    // objective
    double energy = 0.0;
    for (int u = 0; u < U; ++u) {
        energy += std::real(R_blocks[u].trace());
    }
    double F = delta.dot(logdet_S) - lambda * energy;
    phi = -F;

    // backward pass: compute gradients
    grad_blocks.resize(U);
    MatrixXcd tail = MatrixXcd::Zero(Ly, Ly);

    for (int u = U - 1; u >= 0; --u) {
        tail += delta(u) * invS[u];

        MatrixXcd Hu =
            Hn.middleCols(idx_start[u], idx_end[u] - idx_start[u] + 1);
        const auto& Bu = B_blocks[u];
        int Lu = Lxu[u];

        MatrixXcd grad_Rxx =
            Hu.adjoint() * tail * Hu - lambda * MatrixXcd::Identity(Lu, Lu);
        grad_Rxx = make_hermitian(grad_Rxx);

        MatrixXcd grad_B = 2.0 * grad_Rxx * Bu;

        grad_blocks[u] = -grad_B;
    }

    return true;
}

void lbfgs_direction(const VectorXd& grad, const std::vector<VectorXd>& S_hist,
                     const std::vector<VectorXd>& Y_hist, VectorXd& dir) {
    const int k = S_hist.size();
    if (k == 0) {
        dir = -grad;
        return;
    }

    VectorXd q = grad;
    VectorXd alpha_ws(k), rho_ws(k);

    for (int i = k - 1; i >= 0; --i) {
        double denom = Y_hist[i].dot(S_hist[i]);
        if (denom <= 1e-12) {
            rho_ws(i) = 0.0;
            alpha_ws(i) = 0.0;
            continue;
        }
        rho_ws(i) = 1.0 / denom;
        alpha_ws(i) = rho_ws(i) * S_hist[i].dot(q);
        q -= alpha_ws(i) * Y_hist[i];
    }

    double yk_yk = Y_hist.back().squaredNorm();
    double H0 = 1.0;
    if (yk_yk > 1e-12) {
        H0 = S_hist.back().dot(Y_hist.back()) / yk_yk;
    }
    VectorXd r = H0 * q;

    for (int i = 0; i < k; ++i) {
        if (rho_ws(i) == 0.0) continue;
        double beta = rho_ws(i) * Y_hist[i].dot(r);
        r += S_hist[i] * (alpha_ws(i) - beta);
    }

    dir = -r;
}

}  // namespace

ToneResult solve_tone(const MatrixXcd& Hn, const std::vector<int>& Lxu,
                      const std::vector<int>& idx_start,
                      const std::vector<int>& idx_end, const VectorXd& delta,
                      double lambda, const LBFGSConfig& cfg) {
    const int Ly = Hn.rows();
    const int U = Lxu.size();

    // initialize B factors: B = sqrt(1e-6)*I
    std::vector<MatrixXcd> B_blocks(U);
    for (int u = 0; u < U; ++u) {
        B_blocks[u] = std::sqrt(1e-6) * MatrixXcd::Identity(Lxu[u], Lxu[u]);
    }

    int total_params = 0;
    for (int u = 0; u < U; ++u) total_params += 2 * Lxu[u] * Lxu[u];

    VectorXd x(total_params), grad(total_params), dir(total_params);

    std::vector<VectorXd> S_hist, Y_hist;

    // initial evaluation
    double phi;
    std::vector<MatrixXcd> grad_blocks, R_blocks;
    pack_blocks(B_blocks, Lxu, x);

    if (!eval_gradobj(B_blocks, Hn, delta, lambda, Lxu, idx_start, idx_end, U,
                      Ly, phi, grad_blocks, R_blocks)) {
        // should not happen with identity init, but handle gracefully
        ToneResult result;
        result.Rxx.resize(U);
        result.E = VectorXd::Zero(U);
        for (int u = 0; u < U; ++u) {
            result.Rxx[u] = MatrixXcd::Zero(Lxu[u], Lxu[u]);
        }
        return result;
    }

    pack_blocks(grad_blocks, Lxu, grad);

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        double grad_norm = grad.norm();
        if (grad_norm < cfg.tol_grad) break;

        lbfgs_direction(grad, S_hist, Y_hist, dir);
        double dir_deriv = grad.dot(dir);

        if (dir_deriv >= 0) {
            dir = -grad;
            dir_deriv = grad.dot(dir);
            S_hist.clear();
            Y_hist.clear();
        }

        // backtracking line search
        double step = cfg.step_init;
        bool accepted = false;
        VectorXd x_new(total_params), grad_new(total_params);
        double phi_new;
        std::vector<MatrixXcd> grad_new_blocks, R_new;

        for (int ls = 0; ls < cfg.max_linesearch; ++ls) {
            x_new = x + step * dir;
            std::vector<MatrixXcd> B_new(U);
            unpack_blocks(x_new, Lxu, B_new);

            if (eval_gradobj(B_new, Hn, delta, lambda, Lxu, idx_start, idx_end,
                             U, Ly, phi_new, grad_new_blocks, R_new)) {
                if (phi_new <= phi + cfg.armijo_c * step * dir_deriv) {
                    accepted = true;
                    break;
                }
            }
            step *= cfg.beta;
            if (step < cfg.step_min) break;
        }

        if (!accepted) {
            // fallback to gradient descent
            dir = -grad;
            dir_deriv = grad.dot(dir);
            step = cfg.step_init;

            for (int ls = 0; ls < cfg.max_linesearch; ++ls) {
                x_new = x + step * dir;
                std::vector<MatrixXcd> B_new(U);
                unpack_blocks(x_new, Lxu, B_new);

                if (eval_gradobj(B_new, Hn, delta, lambda, Lxu, idx_start,
                                 idx_end, U, Ly, phi_new, grad_new_blocks,
                                 R_new)) {
                    if (phi_new <= phi + cfg.armijo_c * step * dir_deriv) {
                        accepted = true;
                        break;
                    }
                }
                step *= cfg.beta;
                if (step < cfg.step_min) break;
            }

            if (!accepted) break;

            S_hist.clear();
            Y_hist.clear();
        }

        // update L-BFGS history
        pack_blocks(grad_new_blocks, Lxu, grad_new);
        VectorXd s = x_new - x;
        VectorXd y = grad_new - grad;
        double sy = s.dot(y);

        if (sy > 1e-12) {
            if (static_cast<int>(S_hist.size()) >= cfg.m_hist) {
                S_hist.erase(S_hist.begin());
                Y_hist.erase(Y_hist.begin());
            }
            S_hist.push_back(s);
            Y_hist.push_back(y);
        } else {
            S_hist.clear();
            Y_hist.clear();
        }

        x = x_new;
        phi = phi_new;
        grad = grad_new;

        unpack_blocks(x, Lxu, B_blocks);
        R_blocks = R_new;
    }

    // extract optimal Rxx and energy
    ToneResult result;
    result.Rxx.resize(U);
    result.E.resize(U);

    for (int u = 0; u < U; ++u) {
        MatrixXcd Ru = B_blocks[u] * B_blocks[u].adjoint();
        result.Rxx[u] = make_hermitian(Ru);
        result.E(u) = std::real(result.Rxx[u].trace());
    }

    return result;
}

}  // namespace maxresmac
