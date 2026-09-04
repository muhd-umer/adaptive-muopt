/**
 * @file maxrmac_mex.cpp
 * @brief MEX gateway for maxRMAC MIMO solver
 *
 * Compile with CMake (recommended) or:
 *   mex -R2018a maxrmac_mex.cpp solve_tone.cpp maxrmac.cpp
 *       -I/path/to/eigen3 CXXFLAGS='$CXXFLAGS -fopenmp -O3'
 *       LDFLAGS='$LDFLAGS -fopenmp'
 *
 * @author Muhammad Umer
 * @organization Stanford University
 */

#include <complex>
#include <cstring>
#include <vector>

#include "matrix.h"
#include "maxrmac.hpp"
#include "mex.h"

using namespace maxrmac;

/**
 * Gateway function: [Rxxs, Eun, w, bun] = maxrmac_mex(H, Lxu, Eu, theta, cb)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5) {
        mexErrMsgIdAndTxt("maxrmac:nrhs",
                          "Five inputs required: H, Lxu, Eu, theta, cb");
    }
    if (nlhs < 4) {
        mexErrMsgIdAndTxt("maxrmac:nlhs",
                          "Four outputs required: Rxxs, Eun, w, bun");
    }

    const mxArray* H_mx = prhs[0];
    if (!mxIsComplex(H_mx) && !mxIsDouble(H_mx)) {
        mexErrMsgIdAndTxt("maxrmac:invalidInput", "H must be a double array");
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
        mexErrMsgIdAndTxt("maxrmac:invalidInput", "H must be 2D or 3D array");
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

    double* Eu_data = mxGetPr(prhs[2]);
    size_t U = mxGetNumberOfElements(prhs[2]);
    VectorXd Eu(U);
    for (size_t i = 0; i < U; ++i) {
        Eu(i) = Eu_data[i];
    }

    double* theta_data = mxGetPr(prhs[3]);
    VectorXd theta(U);
    for (size_t i = 0; i < U; ++i) {
        theta(i) = theta_data[i];
    }

    int cb = static_cast<int>(mxGetScalar(prhs[4]));

    if (Lxu.size() == 1) {
        int Lxu_scalar = Lxu[0];
        Lxu.resize(U, Lxu_scalar);
    }

    MaxRMACConfig config;
    MaxRMACResult result = maxRMACMIMO(H, Lxu, Eu, theta, cb, config);

    const int N = result.bun.cols();

    plhs[0] = mxCreateCellMatrix(U, N);
    for (int u = 0; u < static_cast<int>(U); ++u) {
        int Lu = Lxu[u];
        for (int n = 0; n < N; ++n) {
            mxArray* Rxx_mx = mxCreateDoubleMatrix(Lu, Lu, mxCOMPLEX);
            mxComplexDouble* Rxx_data = mxGetComplexDoubles(Rxx_mx);

            for (int j = 0; j < Lu; ++j) {
                for (int i = 0; i < Lu; ++i) {
                    size_t idx = i + j * Lu;
                    Rxx_data[idx].real = std::real(result.Rxx[u][n](i, j));
                    Rxx_data[idx].imag = std::imag(result.Rxx[u][n](i, j));
                }
            }

            mxSetCell(plhs[0], u + U * n, Rxx_mx);
        }
    }

    plhs[1] = mxCreateDoubleMatrix(U, N, mxREAL);
    double* Eun_out = mxGetPr(plhs[1]);
    for (int n = 0; n < N; ++n) {
        for (int u = 0; u < static_cast<int>(U); ++u) {
            Eun_out[u + U * n] = result.Eun(u, n);
        }
    }

    plhs[2] = mxCreateDoubleMatrix(U, 1, mxREAL);
    double* w_out = mxGetPr(plhs[2]);
    for (int u = 0; u < static_cast<int>(U); ++u) {
        w_out[u] = result.w(u);
    }

    plhs[3] = mxCreateDoubleMatrix(U, N, mxREAL);
    double* bun_out = mxGetPr(plhs[3]);
    for (int n = 0; n < N; ++n) {
        for (int u = 0; u < static_cast<int>(U); ++u) {
            bun_out[u + U * n] = result.bun(u, n);
        }
    }
}
