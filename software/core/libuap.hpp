// libuap: performance core for OpenUSCT.
//
// Header-only C++ implementation of the acoustic forward model and the exact
// discrete adjoint-state FWI gradient, mirroring the Python reference in
// simulation/ringfwi so the two agree to machine precision. Dimension-general
// (2D and 3D). Both the pybind11 (Python) and MEX (MATLAB) bindings include
// this header, so there is a single source of truth for the numerics.
//
// Grid indices are passed as flat linear indices (row-major: 2D iy*nx+ix,
// 3D (iz*ny+iy)*nx+ix), which the language bindings compute from element
// positions. This keeps the core free of any tuple or geometry handling.

#pragma once
#include <cmath>
#include <vector>
#include <cstddef>

namespace uap {

// Number of grid cells for a shape given as dims[0..ndim-1].
inline std::size_t cell_count(const int* dims, int ndim) {
    std::size_t n = 1;
    for (int d = 0; d < ndim; ++d) n *= static_cast<std::size_t>(dims[d]);
    return n;
}

// Second-order finite-difference Laplacian. Zeroes the whole field, then fills
// interior cells (a one-cell border stays zero), matching the Python stencil.
inline void laplacian(const double* p, double* out, const int* dims, int ndim, double inv_h2) {
    const std::size_t N = cell_count(dims, ndim);
    for (std::size_t i = 0; i < N; ++i) out[i] = 0.0;

    if (ndim == 2) {
        const int ny = dims[0], nx = dims[1];
        #pragma omp parallel for schedule(static)
        for (int iy = 1; iy < ny - 1; ++iy) {
            for (int ix = 1; ix < nx - 1; ++ix) {
                const std::size_t c = static_cast<std::size_t>(iy) * nx + ix;
                out[c] = (p[c + 1] + p[c - 1] + p[c + nx] + p[c - nx] - 4.0 * p[c]) * inv_h2;
            }
        }
    } else {  // ndim == 3
        const int nz = dims[0], ny = dims[1], nx = dims[2];
        const std::size_t syz = static_cast<std::size_t>(ny) * nx;
        #pragma omp parallel for schedule(static)
        for (int iz = 1; iz < nz - 1; ++iz) {
            for (int iy = 1; iy < ny - 1; ++iy) {
                for (int ix = 1; ix < nx - 1; ++ix) {
                    const std::size_t c = static_cast<std::size_t>(iz) * syz + static_cast<std::size_t>(iy) * nx + ix;
                    out[c] = (p[c + 1] + p[c - 1] + p[c + nx] + p[c - nx] + p[c + syz] + p[c - syz]
                              - 6.0 * p[c]) * inv_h2;
                }
            }
        }
    }
}

// One transmit event: leapfrog the wave equation, record receiver traces, and
// optionally store the full pressure history. Layout: p^n at time level n,
// with the source sample used when advancing to level n being wavelet[n-1].
//   rec  : nt * n_rec  (row-major, rec[n*n_rec + j]); may be null
//   hist : nt * N      (row-major, hist[n*N + i]);     may be null
inline void forward(const double* m, const int* dims, int ndim, double h, double dt, int nt,
                    int src_lin, const double* wavelet,
                    const int* rec_lin, int n_rec,
                    double* rec, double* hist) {
    const std::size_t N = cell_count(dims, ndim);
    const long long Nll = static_cast<long long>(N);
    const double inv_h2 = 1.0 / (h * h);
    const double dt2 = dt * dt;

    std::vector<double> S(N), a(N, 0.0), b(N, 0.0), c(N, 0.0), lap(N, 0.0);
    for (std::size_t i = 0; i < N; ++i) S[i] = dt2 / m[i];

    double* p_prev = a.data();
    double* p_cur = b.data();
    double* p_new = c.data();

    // Level 0 is zero.
    if (hist) for (std::size_t i = 0; i < N; ++i) hist[i] = 0.0;
    if (rec) for (int j = 0; j < n_rec; ++j) rec[j] = 0.0;

    for (int n = 1; n < nt; ++n) {
        laplacian(p_cur, lap.data(), dims, ndim, inv_h2);
        lap[static_cast<std::size_t>(src_lin)] += wavelet[n - 1];
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < Nll; ++i)
            p_new[i] = 2.0 * p_cur[i] - p_prev[i] + S[i] * lap[i];

        double* tmp = p_prev;  // rotate: prev<-cur, cur<-new, new<-old prev
        p_prev = p_cur;
        p_cur = p_new;
        p_new = tmp;

        if (hist) {
            double* row = hist + static_cast<std::size_t>(n) * N;
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < Nll; ++i) row[i] = p_cur[i];
        }
        if (rec) {
            double* row = rec + static_cast<std::size_t>(n) * n_rec;
            for (int j = 0; j < n_rec; ++j) row[j] = p_cur[static_cast<std::size_t>(rec_lin[j])];
        }
    }
}

// Full-matrix-capture forward: each transmitter fires; every receiver records.
//   data : n_tx * nt * n_rec (row-major)
inline void forward_fmc(const double* m, const int* dims, int ndim, double h, double dt, int nt,
                        const int* tx_lin, int n_tx, const int* rec_lin, int n_rec,
                        const double* wavelet, double* data) {
    for (int t = 0; t < n_tx; ++t) {
        double* rec = data + static_cast<std::size_t>(t) * nt * n_rec;
        forward(m, dims, ndim, h, dt, nt, tx_lin[t], wavelet, rec_lin, n_rec, rec, nullptr);
    }
}

// Misfit value and adjoint source for one transmit's data (nt x n_rec,
// time-major). misfit_type 0 = least squares (adjoint source = residual);
// misfit_type 1 = global correlation norm (per-trace normalised
// cross-correlation, J = sum_j (1 - c_j)), matching ringfwi.fwi.
inline double adjoint_residual(const double* dsyn, const double* dobs,
                               int nt, int n_rec, int misfit_type, double* res) {
    const std::size_t total = static_cast<std::size_t>(nt) * n_rec;
    if (misfit_type == 0) {
        double J = 0.0;
        for (std::size_t k = 0; k < total; ++k) {
            res[k] = dsyn[k] - dobs[k];
            J += 0.5 * res[k] * res[k];
        }
        return J;
    }
    // GCN: per-trace (per receiver column) norms and correlation.
    const double eps = 1e-12;
    std::vector<double> ns(n_rec, 0.0), no(n_rec, 0.0), c(n_rec, 0.0);
    for (int k = 0; k < nt; ++k) {
        const double* srow = dsyn + static_cast<std::size_t>(k) * n_rec;
        const double* orow = dobs + static_cast<std::size_t>(k) * n_rec;
        for (int j = 0; j < n_rec; ++j) {
            ns[j] += srow[j] * srow[j];
            no[j] += orow[j] * orow[j];
        }
    }
    for (int j = 0; j < n_rec; ++j) {
        ns[j] = std::sqrt(ns[j]) + eps;
        no[j] = std::sqrt(no[j]) + eps;
    }
    for (int k = 0; k < nt; ++k) {
        const double* srow = dsyn + static_cast<std::size_t>(k) * n_rec;
        const double* orow = dobs + static_cast<std::size_t>(k) * n_rec;
        for (int j = 0; j < n_rec; ++j) c[j] += (srow[j] / ns[j]) * (orow[j] / no[j]);
    }
    double J = 0.0;
    for (int j = 0; j < n_rec; ++j) J += 1.0 - c[j];
    for (int k = 0; k < nt; ++k) {
        const double* srow = dsyn + static_cast<std::size_t>(k) * n_rec;
        const double* orow = dobs + static_cast<std::size_t>(k) * n_rec;
        double* rrow = res + static_cast<std::size_t>(k) * n_rec;
        for (int j = 0; j < n_rec; ++j)
            rrow[j] = -(orow[j] / no[j] - c[j] * (srow[j] / ns[j])) / ns[j];
    }
    return J;
}

// Waveform misfit and its exact discrete-adjoint gradient in the squared-
// slowness field. dobs layout matches forward_fmc; grad has N cells.
// misfit_type: 0 = least squares, 1 = global correlation norm.
// Returns the misfit J.
inline double misfit_and_gradient(const double* m, const int* dims, int ndim,
                                  double h, double dt, int nt,
                                  const int* tx_lin, int n_tx, const int* rec_lin, int n_rec,
                                  const double* wavelet, const double* dobs, double* grad,
                                  int misfit_type = 0) {
    const std::size_t N = cell_count(dims, ndim);
    const long long Nll = static_cast<long long>(N);
    const double inv_h2 = 1.0 / (h * h);
    const double dt2 = dt * dt;

    std::vector<double> S(N);
    for (std::size_t i = 0; i < N; ++i) S[i] = dt2 / m[i];
    for (std::size_t i = 0; i < N; ++i) grad[i] = 0.0;

    std::vector<double> U(static_cast<std::size_t>(nt) * N);
    std::vector<double> dsyn(static_cast<std::size_t>(nt) * n_rec);
    std::vector<double> res(static_cast<std::size_t>(nt) * n_rec);
    std::vector<double> lam_p1(N), lam_p2(N), lam_k(N), tmp(N), lap(N);

    double J = 0.0;

    for (int t = 0; t < n_tx; ++t) {
        forward(m, dims, ndim, h, dt, nt, tx_lin[t], wavelet, rec_lin, n_rec, dsyn.data(), U.data());

        const double* dob = dobs + static_cast<std::size_t>(t) * nt * n_rec;
        J += adjoint_residual(dsyn.data(), dob, nt, n_rec, misfit_type, res.data());

        for (std::size_t i = 0; i < N; ++i) { lam_p1[i] = 0.0; lam_p2[i] = 0.0; }

        for (int k = nt - 1; k >= 1; --k) {
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < Nll; ++i) tmp[i] = S[i] * lam_p1[i];
            laplacian(tmp.data(), lap.data(), dims, ndim, inv_h2);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < Nll; ++i) lam_k[i] = 2.0 * lam_p1[i] - lam_p2[i] + lap[i];

            const double* rrow = res.data() + static_cast<std::size_t>(k) * n_rec;
            for (int j = 0; j < n_rec; ++j) lam_k[static_cast<std::size_t>(rec_lin[j])] -= rrow[j];

            const double* Uk = U.data() + static_cast<std::size_t>(k) * N;
            const double* Uk1 = U.data() + static_cast<std::size_t>(k - 1) * N;
            const double* Uk2 = (k >= 2) ? U.data() + static_cast<std::size_t>(k - 2) * N : nullptr;
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < Nll; ++i) {
                double d2p = Uk[i] - 2.0 * Uk1[i] + (Uk2 ? Uk2[i] : 0.0);
                grad[i] += lam_k[i] * d2p;
            }

            lam_p2.swap(lam_p1);
            lam_p1.swap(lam_k);
        }
    }

    for (std::size_t i = 0; i < N; ++i) grad[i] /= m[i];
    return J;
}

}  // namespace uap
