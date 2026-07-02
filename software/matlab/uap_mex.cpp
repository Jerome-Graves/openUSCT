// MATLAB MEX gateway for the libuap C++ core.
//
// Kept deliberately simple: it operates on flat, row-major arrays and explicit
// dimensions, so it shares libuap.hpp unchanged with the Python binding. The
// column-major <-> row-major layout conversion lives in the .m wrappers
// (uap_forward_fmc.m, uap_misfit_and_gradient.m).
//
// Usage (from the wrappers):
//   data       = uap_mex('forward',  m_flat, dims, h, dt, nt, tx_lin, rec_lin, wavelet)
//   [J, grad]  = uap_mex('gradient', m_flat, dims, h, dt, nt, tx_lin, rec_lin, wavelet, dobs)
// where m_flat is row-major, and tx_lin/rec_lin are 0-based row-major indices.

#include "mex.h"
#include <string>
#include <vector>

#include "libuap.hpp"

static std::vector<int> to_int(const mxArray* a) {
    const double* p = mxGetDoubles(a);
    size_t n = mxGetNumberOfElements(a);
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int>(p[i]);
    return v;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 9) mexErrMsgTxt("uap_mex: need at least 9 arguments");

    char mode[16];
    mxGetString(prhs[0], mode, sizeof(mode));

    const double* m = mxGetDoubles(prhs[1]);
    std::vector<int> dims = to_int(prhs[2]);
    int ndim = static_cast<int>(dims.size());
    double h = mxGetScalar(prhs[3]);
    double dt = mxGetScalar(prhs[4]);
    int nt = static_cast<int>(mxGetScalar(prhs[5]));
    std::vector<int> tx = to_int(prhs[6]);
    std::vector<int> rx = to_int(prhs[7]);
    const double* wavelet = mxGetDoubles(prhs[8]);
    int n_tx = static_cast<int>(tx.size());
    int n_rec = static_cast<int>(rx.size());

    if (std::string(mode) == "forward") {
        mwSize len = static_cast<mwSize>(n_tx) * nt * n_rec;
        plhs[0] = mxCreateDoubleMatrix(len, 1, mxREAL);
        uap::forward_fmc(m, dims.data(), ndim, h, dt, nt, tx.data(), n_tx,
                         rx.data(), n_rec, wavelet, mxGetDoubles(plhs[0]));
    } else if (std::string(mode) == "gradient") {
        if (nrhs < 10) mexErrMsgTxt("uap_mex gradient: need dobs as 10th argument");
        const double* dobs = mxGetDoubles(prhs[9]);
        size_t N = uap::cell_count(dims.data(), ndim);
        plhs[0] = mxCreateDoubleScalar(0.0);
        plhs[1] = mxCreateDoubleMatrix(static_cast<mwSize>(N), 1, mxREAL);
        double J = uap::misfit_and_gradient(m, dims.data(), ndim, h, dt, nt,
                                            tx.data(), n_tx, rx.data(), n_rec,
                                            wavelet, dobs, mxGetDoubles(plhs[1]));
        *mxGetDoubles(plhs[0]) = J;
    } else {
        mexErrMsgTxt("uap_mex: mode must be 'forward' or 'gradient'");
    }
}
