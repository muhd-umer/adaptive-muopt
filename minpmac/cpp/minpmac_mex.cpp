/**
 * @file minpmac_mex.cpp
 * @brief MEX gateway for minPMAC MIMO solver
 *
 * Compile with CMake (recommended) or:
 *   mex -R2018a minpmac_mex.cpp solve_tone.cpp minpmac.cpp
 *       -I/path/to/eigen3 -lglpk CXXFLAGS='$CXXFLAGS -fopenmp -O3'
 *       LDFLAGS='$LDFLAGS -fopenmp'
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include <complex>
#include <cstring>
#include <vector>

#include "matrix.h"
#include "mex.h"
#include "minpmac.hpp"

using namespace minpmac;

/**
 * Gateway function: [feas_flag, bu_a, info] = minpmac_mex(H, Lxu, bu_min, w,
 * cb)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5) {
        mexErrMsgIdAndTxt("minpmac:nrhs",
                          "Five inputs required: H, Lxu, bu_min, w, cb");
    }
    if (nlhs < 3) {
        mexErrMsgIdAndTxt("minpmac:nlhs",
                          "Three outputs required: feas_flag, bu_a, info");
    }

    const mxArray* H_mx = prhs[0];
    if (!mxIsComplex(H_mx) && !mxIsDouble(H_mx)) {
        mexErrMsgIdAndTxt("minpmac:invalidInput", "H must be a double array");
    }

    const mwSize* H_dims = mxGetDimensions(H_mx);
    mwSize ndims = mxGetNumberOfDimensions(H_mx);

    int Ly, Ltot, N_in;
    if (ndims == 2) {
        Ly = static_cast<int>(H_dims[0]);
        Ltot = static_cast<int>(H_dims[1]);
        N_in = 1;
    } else if (ndims == 3) {
        Ly = static_cast<int>(H_dims[0]);
        Ltot = static_cast<int>(H_dims[1]);
        N_in = static_cast<int>(H_dims[2]);
    } else {
        mexErrMsgIdAndTxt("minpmac:invalidInput", "H must be 2D or 3D array");
    }

    std::vector<MatrixXcd> H(N_in);

    if (mxIsComplex(H_mx)) {
        mxComplexDouble* H_data = mxGetComplexDoubles(H_mx);
        for (int n = 0; n < N_in; ++n) {
            H[n].resize(Ly, Ltot);
            for (int j = 0; j < Ltot; ++j) {
                for (int i = 0; i < Ly; ++i) {
                    size_t idx = i + j * Ly + n * Ly * Ltot;
                    H[n](i, j) = Complex(H_data[idx].real, H_data[idx].imag);
                }
            }
        }
    } else {
        double* H_data = mxGetDoubles(H_mx);
        for (int n = 0; n < N_in; ++n) {
            H[n].resize(Ly, Ltot);
            for (int j = 0; j < Ltot; ++j) {
                for (int i = 0; i < Ly; ++i) {
                    size_t idx = i + j * Ly + n * Ly * Ltot;
                    H[n](i, j) = Complex(H_data[idx], 0.0);
                }
            }
        }
    }

    double* Lxu_data = mxGetPr(prhs[1]);
    size_t Lxu_len = mxGetNumberOfElements(prhs[1]);
    std::vector<int> Lxu(Lxu_len);
    for (size_t i = 0; i < Lxu_len; ++i) {
        Lxu[i] = static_cast<int>(Lxu_data[i]);
    }

    double* bu_min_data = mxGetPr(prhs[2]);
    size_t U = mxGetNumberOfElements(prhs[2]);
    VectorXd bu_min(U);
    for (size_t i = 0; i < U; ++i) {
        bu_min(i) = bu_min_data[i];
    }

    double* w_data = mxGetPr(prhs[3]);
    VectorXd w(U);
    for (size_t i = 0; i < U; ++i) {
        w(i) = w_data[i];
    }

    int cb = static_cast<int>(mxGetScalar(prhs[4]));

    if (Lxu.size() == 1) {
        int Lxu_scalar = Lxu[0];
        Lxu.resize(U, Lxu_scalar);
    }

    MinPMACConfig config;
    MinPMACResult result = minPMACMIMO(H, Lxu, bu_min, w, cb, config);

    const int N = result.bun.cols();

    plhs[0] = mxCreateDoubleScalar(static_cast<double>(result.feas_flag));

    plhs[1] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* bu_a_out = mxGetPr(plhs[1]);
    for (size_t u = 0; u < U; ++u) {
        bu_a_out[u] = result.bu_a(u);
    }

    const char* field_names[] = {
        "bun", "frac",       "Eu_avg",       "theta",      "orderings",
        "Rxx", "iterations", "elapsed_time", "sol_status", "feasible"};
    mxArray* info_struct = mxCreateStructMatrix(1, 1, 10, field_names);

    mxArray* bun_mx = mxCreateDoubleMatrix(U, N, mxREAL);
    double* bun_out = mxGetPr(bun_mx);
    for (int n = 0; n < N; ++n) {
        for (size_t u = 0; u < U; ++u) {
            bun_out[u + U * n] = result.bun(u, n);
        }
    }
    mxSetField(info_struct, 0, "bun", bun_mx);

    size_t num_frac = result.frac.size();
    mxArray* frac_mx;
    if (num_frac > 0) {
        frac_mx = mxCreateDoubleMatrix(num_frac, 1, mxREAL);
        double* frac_out = mxGetPr(frac_mx);
        for (size_t i = 0; i < num_frac; ++i) {
            frac_out[i] = result.frac(i);
        }
    } else {
        frac_mx = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
    mxSetField(info_struct, 0, "frac", frac_mx);

    mxArray* Eu_avg_mx = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* Eu_avg_out = mxGetPr(Eu_avg_mx);
    for (size_t u = 0; u < U; ++u) {
        Eu_avg_out[u] = result.Eu_avg(u);
    }
    mxSetField(info_struct, 0, "Eu_avg", Eu_avg_mx);

    mxArray* theta_mx = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* theta_out = mxGetPr(theta_mx);
    for (size_t u = 0; u < U; ++u) {
        theta_out[u] = result.theta(u);
    }
    mxSetField(info_struct, 0, "theta", theta_mx);

    size_t num_orderings = result.orderings.size();
    mxArray* orderings_mx = mxCreateCellMatrix(num_orderings, 1);
    for (size_t k = 0; k < num_orderings; ++k) {
        const auto& order = result.orderings[k];
        mxArray* order_mx = mxCreateDoubleMatrix(1, order.size(), mxREAL);
        double* order_out = mxGetPr(order_mx);
        for (size_t i = 0; i < order.size(); ++i) {
            order_out[i] = static_cast<double>(order[i] + 1);
        }
        mxSetCell(orderings_mx, k, order_mx);
    }
    mxSetField(info_struct, 0, "orderings", orderings_mx);

    if (!result.Rxx.empty()) {
        const int Rxx_rows = result.Rxx[0].rows();
        const int Rxx_cols = result.Rxx[0].cols();
        const size_t num_tones = result.Rxx.size();
        mwSize Rxx_dims[3] = {static_cast<mwSize>(Rxx_rows),
                              static_cast<mwSize>(Rxx_cols),
                              static_cast<mwSize>(num_tones)};
        mxArray* Rxx_mx =
            mxCreateNumericArray(3, Rxx_dims, mxDOUBLE_CLASS, mxCOMPLEX);
        mxComplexDouble* Rxx_out = mxGetComplexDoubles(Rxx_mx);
        for (size_t n = 0; n < num_tones; ++n) {
            const auto& Rn = result.Rxx[n];
            for (int j = 0; j < Rxx_cols; ++j) {
                for (int i = 0; i < Rxx_rows; ++i) {
                    size_t idx = i + j * Rxx_rows + n * Rxx_rows * Rxx_cols;
                    Rxx_out[idx].real = Rn(i, j).real();
                    Rxx_out[idx].imag = Rn(i, j).imag();
                }
            }
        }
        mxSetField(info_struct, 0, "Rxx", Rxx_mx);
    } else {
        mxSetField(info_struct, 0, "Rxx",
                   mxCreateDoubleMatrix(0, 0, mxCOMPLEX));
    }

    mxSetField(info_struct, 0, "iterations",
               mxCreateDoubleScalar(static_cast<double>(result.iterations)));

    mxSetField(info_struct, 0, "elapsed_time",
               mxCreateDoubleScalar(result.elapsed_sec));

    mxArray* sol_status_mx =
        mxCreateString(result.feas_flag > 0 ? "Solved" : "Failed");
    mxSetField(info_struct, 0, "sol_status", sol_status_mx);

    mxSetField(info_struct, 0, "feasible",
               mxCreateLogicalScalar(result.feas_flag > 0));

    plhs[2] = info_struct;
}
