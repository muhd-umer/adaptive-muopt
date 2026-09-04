/**
 * @file solve_tone.cpp
 * @brief Per-tone L-BFGS optimization implementation for minPMAC
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include "solve_tone.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minpmac {

namespace {

void pack_blocks(const std::vector<MatrixXcd>& blocks,
                 const std::vector<int>& block_sizes, VectorXd& vec) {
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

void unpack_blocks(const VectorXd& vec, const std::vector<int>& block_sizes,
                   std::vector<MatrixXcd>& blocks) {
    int offset = 0;
    for (size_t u = 0; u < block_sizes.size(); ++u) {
        const int Lu = block_sizes[u];
        const int n = Lu * Lu;
        blocks[u].resize(Lu, Lu);
        for (int i = 0; i < n; ++i) {
            blocks[u].data()[i] = Complex(vec(offset + i), vec(offset + n + i));
        }
        offset += 2 * n;
    }
}

/**
 * @brief Evaluate objective and gradient for L-BFGS
 *
 * Objective: F = sum_k alpha_k * log|S_k| - sum_u w_u * ||B_u||^2
 * where S_k = I + sum_{j=1}^k H_j * B_j * B_j' * H_j'
 *
 * We minimize phi = -F
 */
bool eval_objective(const VectorXd& x, const std::vector<MatrixXcd>& H_blocks,
                    const VectorXd& alpha, const VectorXd& sw,
                    const std::vector<int>& block_sizes, int Ly,
                    ToneWorkspace& ws, double& phi, VectorXd& grad_vec) {
    const int U = H_blocks.size();

    unpack_blocks(x, block_sizes, ws.B_blocks);

    ws.S = MatrixXcd::Identity(Ly, Ly);
    double energy_term = 0.0;

    for (int u = 0; u < U; ++u) {
        const auto& Hu = H_blocks[u];
        const auto& Bu = ws.B_blocks[u];

        ws.R_blocks[u] = Bu * Bu.adjoint();
        ws.R_blocks[u] = make_hermitian(ws.R_blocks[u]);

        ws.HRH.noalias() = Hu * ws.R_blocks[u] * Hu.adjoint();
        ws.S += ws.HRH;
        ws.S = make_hermitian(ws.S);

        double logdet;
        if (!cholesky_with_inv(ws.S, ws.L, ws.invS[u], logdet)) {
            phi = std::numeric_limits<double>::infinity();
            return false;
        }

        ws.logdet_S(u) = logdet;

        energy_term += sw(u) * ws.B_blocks[u].squaredNorm();
    }

    double F = alpha.dot(ws.logdet_S) - energy_term;
    phi = -F;

    ws.tail.setZero(Ly, Ly);

    int offset = 0;
    for (int u = 0; u < U; ++u) {
        offset += 2 * block_sizes[u] * block_sizes[u];
    }

    for (int u = U - 1; u >= 0; --u) {
        const int Lu = block_sizes[u];
        const int n = Lu * Lu;
        offset -= 2 * n;

        ws.tail += alpha(u) * ws.invS[u];

        const auto& Hu = H_blocks[u];
        const auto& Bu = ws.B_blocks[u];

        ws.grad_R.noalias() = Hu.adjoint() * ws.tail * Hu;
        ws.grad_R -= sw(u) * MatrixXcd::Identity(Lu, Lu);
        ws.grad_R = make_hermitian(ws.grad_R);

        ws.grad_B.noalias() = 2.0 * ws.grad_R * Bu;

        for (int i = 0; i < n; ++i) {
            grad_vec(offset + i) = -std::real(ws.grad_B.data()[i]);
            grad_vec(offset + n + i) = -std::imag(ws.grad_B.data()[i]);
        }
    }

    return true;
}

/**
 * @brief Compute L-BFGS search direction
 */
void lbfgs_direction(const VectorXd& grad, const std::vector<VectorXd>& S_hist,
                     const std::vector<VectorXd>& Y_hist, VectorXd& alpha_ws,
                     VectorXd& rho_ws, VectorXd& q_ws, VectorXd& dir) {
    const int k = S_hist.size();

    if (k == 0) {
        dir = -grad;
        return;
    }

    q_ws = grad;

    for (int i = k - 1; i >= 0; --i) {
        const double denom = Y_hist[i].dot(S_hist[i]);
        if (denom <= 1e-12) {
            rho_ws(i) = 0.0;
            alpha_ws(i) = 0.0;
            continue;
        }
        rho_ws(i) = 1.0 / denom;
        alpha_ws(i) = rho_ws(i) * S_hist[i].dot(q_ws);
        q_ws -= alpha_ws(i) * Y_hist[i];
    }

    const double yk_yk = Y_hist.back().squaredNorm();
    double H0 = 1.0;
    if (yk_yk > 1e-12) {
        H0 = S_hist.back().dot(Y_hist.back()) / yk_yk;
    }

    dir = H0 * q_ws;

    for (int i = 0; i < k; ++i) {
        if (rho_ws(i) == 0.0) continue;
        const double beta = rho_ws(i) * Y_hist[i].dot(dir);
        dir += S_hist[i] * (alpha_ws(i) - beta);
    }

    dir = -dir;
}

}  // namespace

ToneResult solve_tone(const MatrixXcd& Hn, const VectorXd& theta,
                      const VectorXd& w, const std::vector<int>& Lxu,
                      const std::vector<int>& idx_start,
                      const std::vector<int>& idx_end, LBFGSState& state_in,
                      ToneWorkspace& ws) {
    const int Ly = Hn.rows();
    const int U = Lxu.size();

    std::vector<int> order(U);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return theta(a) > theta(b); });

    VectorXd stheta(U), sw(U);
    std::vector<int> sLxu(U);
    for (int i = 0; i < U; ++i) {
        stheta(i) = theta(order[i]);
        sw(i) = w(order[i]);
        sLxu[i] = Lxu[order[i]];
    }

    ws.H_blocks.resize(U);
    for (int u = 0; u < U; ++u) {
        int orig = order[u];
        ws.H_blocks[u] =
            Hn.middleCols(idx_start[orig], idx_end[orig] - idx_start[orig] + 1);
    }

    VectorXd alpha(U);
    for (int u = 0; u < U - 1; ++u) {
        alpha(u) = 0.5 * (stheta(u) - stheta(u + 1));
    }
    alpha(U - 1) = 0.5 * stheta(U - 1);

    bool state_valid = state_in.valid && state_in.order == order &&
                       state_in.block_sizes == Lxu;

    int total_params = 0;
    for (int u = 0; u < U; ++u) {
        total_params += 2 * sLxu[u] * sLxu[u];
    }

    ws.B_blocks.resize(U);
    ws.R_blocks.resize(U);

    if (state_valid) {
        ws.B_blocks = state_in.B_blocks;
    } else {
        for (int u = 0; u < U; ++u) {
            ws.B_blocks[u] =
                std::sqrt(1e-6) * MatrixXcd::Identity(sLxu[u], sLxu[u]);
        }
        state_in.S_hist.clear();
        state_in.Y_hist.clear();
    }

    ws.x.resize(total_params);
    ws.x_new.resize(total_params);
    ws.x_trial.resize(total_params);
    ws.grad.resize(total_params);
    ws.grad_new.resize(total_params);
    ws.grad_trial.resize(total_params);
    ws.dir.resize(total_params);
    ws.logdet_S.resize(U);
    ws.invS.resize(U);
    for (int u = 0; u < U; ++u) {
        ws.invS[u].resize(Ly, Ly);
    }

    LBFGSConfig cfg;
    cfg.max_iter = 100;
    cfg.m_hist = 8;
    cfg.tol_grad = 1e-5;

    pack_blocks(ws.B_blocks, sLxu, ws.x);

    double phi;
    if (!eval_objective(ws.x, ws.H_blocks, alpha, sw, sLxu, Ly, ws, phi,
                        ws.grad)) {
        for (int u = 0; u < U; ++u) {
            ws.B_blocks[u] =
                std::sqrt(1e-6) * MatrixXcd::Identity(sLxu[u], sLxu[u]);
        }
        pack_blocks(ws.B_blocks, sLxu, ws.x);
        state_in.S_hist.clear();
        state_in.Y_hist.clear();
        eval_objective(ws.x, ws.H_blocks, alpha, sw, sLxu, Ly, ws, phi,
                       ws.grad);
    }

    ws.alpha.resize(cfg.m_hist);
    ws.rho.resize(cfg.m_hist);
    ws.q.resize(total_params);

    double grad_norm_sq = ws.grad.squaredNorm();
    const double tol_grad_sq = cfg.tol_grad * cfg.tol_grad;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        if (grad_norm_sq < tol_grad_sq) {
            break;
        }

        lbfgs_direction(ws.grad, state_in.S_hist, state_in.Y_hist, ws.alpha,
                        ws.rho, ws.q, ws.dir);

        double dir_deriv = ws.grad.dot(ws.dir);

        if (dir_deriv >= 0) {
            ws.dir = -ws.grad;
            dir_deriv = ws.grad.dot(ws.dir);
            state_in.S_hist.clear();
            state_in.Y_hist.clear();
        }

        double step = cfg.step_init;
        bool accepted = false;
        double phi_new = phi;

        for (int ls = 0; ls < cfg.max_linesearch; ++ls) {
            ws.x_trial = ws.x + step * ws.dir;

            double phi_trial;
            if (eval_objective(ws.x_trial, ws.H_blocks, alpha, sw, sLxu, Ly, ws,
                               phi_trial, ws.grad_trial)) {
                if (phi_trial <= phi + cfg.armijo_c * step * dir_deriv) {
                    accepted = true;
                    ws.x_new = ws.x_trial;
                    phi_new = phi_trial;
                    ws.grad_new = ws.grad_trial;
                    break;
                }
            }

            step *= cfg.beta;
            if (step < cfg.step_min) break;
        }

        if (!accepted) {
            ws.dir = -ws.grad;
            dir_deriv = ws.grad.dot(ws.dir);
            step = cfg.step_init;

            for (int ls = 0; ls < cfg.max_linesearch; ++ls) {
                ws.x_trial = ws.x + step * ws.dir;

                double phi_trial;
                if (eval_objective(ws.x_trial, ws.H_blocks, alpha, sw, sLxu, Ly,
                                   ws, phi_trial, ws.grad_trial)) {
                    if (phi_trial <= phi + cfg.armijo_c * step * dir_deriv) {
                        accepted = true;
                        ws.x_new = ws.x_trial;
                        phi_new = phi_trial;
                        ws.grad_new = ws.grad_trial;
                        break;
                    }
                }

                step *= cfg.beta;
                if (step < cfg.step_min) break;
            }

            if (!accepted) break;

            state_in.S_hist.clear();
            state_in.Y_hist.clear();
        }

        VectorXd s = ws.x_new - ws.x;
        VectorXd y = ws.grad_new - ws.grad;
        double sy = s.dot(y);

        if (sy > 1e-12) {
            if (static_cast<int>(state_in.S_hist.size()) >= cfg.m_hist) {
                state_in.S_hist.erase(state_in.S_hist.begin());
                state_in.Y_hist.erase(state_in.Y_hist.begin());
            }
            state_in.S_hist.push_back(s);
            state_in.Y_hist.push_back(y);
        } else {
            state_in.S_hist.clear();
            state_in.Y_hist.clear();
        }

        ws.x = ws.x_new;
        phi = phi_new;
        ws.grad = ws.grad_new;
        grad_norm_sq = ws.grad.squaredNorm();
    }

    unpack_blocks(ws.x, sLxu, ws.B_blocks);
    for (int u = 0; u < U; ++u) {
        ws.R_blocks[u] = ws.B_blocks[u] * ws.B_blocks[u].adjoint();
        ws.R_blocks[u] = make_hermitian(ws.R_blocks[u]);
    }

    VectorXd b_sorted(U);
    MatrixXcd S_curr = MatrixXcd::Identity(Ly, Ly);
    double logdet_prev = 0.0;

    for (int u = 0; u < U; ++u) {
        S_curr += ws.H_blocks[u] * ws.R_blocks[u] * ws.H_blocks[u].adjoint();
        S_curr = make_hermitian(S_curr);
        double logdet_curr = logdet_spd(S_curr);
        b_sorted(u) = 0.5 * (logdet_curr - logdet_prev);
        logdet_prev = logdet_curr;
    }

    VectorXd E_sorted(U);
    for (int u = 0; u < U; ++u) {
        E_sorted(u) = std::real(ws.R_blocks[u].trace());
    }

    ToneResult result;
    result.b.resize(U);
    result.E.resize(U);
    result.Rxx.resize(U);

    for (int u = 0; u < U; ++u) {
        int orig = order[u];
        result.b(orig) = b_sorted(u);
        result.E(orig) = E_sorted(u);
        result.Rxx[orig] = ws.R_blocks[u];
    }

    result.f = theta.dot(result.b) - w.dot(result.E);

    state_in.B_blocks = ws.B_blocks;
    state_in.order = order;
    state_in.block_sizes = Lxu;
    state_in.valid = true;

    return result;
}

}  // namespace minpmac
