"""uap_gpu: CuPy GPU backend for the FWI forward model and adjoint gradient.

Drop-in `backend` for ringfwi.fwi.invert, running the exact discrete scheme on
the GPU in single precision (consumer GPUs throttle FP64, so FP32 is where they
win; it still agrees with the FP64 CPU reference to the tolerance FWI needs).

The 3D path uses fused CUDA stencil kernels (one launch per time step instead of
about ten), so it is compute-bound rather than kernel-launch-bound. The 2D path
uses plain CuPy array ops.

    import uap_gpu
    fwi.invert(..., backend=uap_gpu)

Requires cupy-cuda12x[ctk] and an NVIDIA GPU.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

DTYPE = cp.float32

# --- fused 3D stencil kernels ----------------------------------------------
# One thread per cell. Interior cells get the 7-point Laplacian; the one-cell
# border gets zero Laplacian, matching the CPU stencil exactly.

_fwd_step = cp.RawKernel(r'''
extern "C" __global__ void fwd_step(const float* pc, const float* pp, const float* S,
                                    float* pn, int nz, int ny, int nx, float inv_h2) {
    long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long N = (long)nz * ny * nx;
    if (c >= N) return;
    int ix = c % nx, iy = (c / nx) % ny, iz = c / ((long)nx * ny);
    float lap = 0.0f;
    if (ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1 && iz > 0 && iz < nz-1) {
        long syz = (long)ny * nx;
        lap = (pc[c+1] + pc[c-1] + pc[c+nx] + pc[c-nx] + pc[c+syz] + pc[c-syz]
               - 6.0f*pc[c]) * inv_h2;
    }
    pn[c] = 2.0f*pc[c] - pp[c] + S[c]*lap;
}
''', 'fwd_step')

_adj_step = cp.RawKernel(r'''
extern "C" __global__ void adj_step(const float* l1, const float* l2, const float* S,
                                    float* lk, int nz, int ny, int nx, float inv_h2) {
    long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long N = (long)nz * ny * nx;
    if (c >= N) return;
    int ix = c % nx, iy = (c / nx) % ny, iz = c / ((long)nx * ny);
    float lap = 0.0f;
    if (ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1 && iz > 0 && iz < nz-1) {
        long syz = (long)ny * nx;
        lap = (S[c+1]*l1[c+1] + S[c-1]*l1[c-1] + S[c+nx]*l1[c+nx] + S[c-nx]*l1[c-nx]
               + S[c+syz]*l1[c+syz] + S[c-syz]*l1[c-syz] - 6.0f*S[c]*l1[c]) * inv_h2;
    }
    lk[c] = 2.0f*l1[c] - l2[c] + lap;
}
''', 'adj_step')

_grad_acc = cp.ElementwiseKernel(
    'float32 lk, float32 uk, float32 uk1, float32 uk2', 'float32 g',
    'g += lk * (uk - 2.0f*uk1 + uk2)', 'grad_acc')

_BLK = 256


def _grid(N):
    return ((N + _BLK - 1) // _BLK,)


# --- dimension-general sliced Laplacian (used by the 2D path) ---------------
def _laplacian(p, inv_h2, out):
    ndim = p.ndim
    out[...] = 0
    inner = tuple(slice(1, -1) for _ in range(ndim))
    acc = (-2.0 * ndim) * p[inner]
    for ax in range(ndim):
        plus = list(inner); minus = list(inner)
        plus[ax] = slice(2, None); minus[ax] = slice(0, -2)
        acc = acc + p[tuple(plus)] + p[tuple(minus)]
    out[inner] = acc * inv_h2


def _lin_indices(geom, shape):
    idx = np.asarray(geom.idx, dtype=np.int64)
    strides = np.cumprod((1,) + tuple(shape[::-1][:-1]))[::-1]
    return cp.asarray((idx * strides).sum(axis=1))


def _axes(geom, ndim):
    idx = np.asarray(geom.idx, dtype=np.int64)
    return tuple(cp.asarray(idx[:, a]) for a in range(ndim))


# ---------------------------------------------------------------------------
# 3D fused path
# ---------------------------------------------------------------------------
def _forward3d(mflat, dims, h, dt, nt, src_lin, wavelet, rec_lin, store):
    nz, ny, nx = dims
    N = mflat.size
    inv_h2 = np.float32(1.0 / (h * h))
    S = (np.float32(dt * dt)) / mflat
    Sw = S[src_lin]
    p_prev = cp.zeros(N, DTYPE); p_cur = cp.zeros(N, DTYPE); p_new = cp.zeros(N, DTYPE)
    rec = cp.zeros((nt, rec_lin.size), DTYPE)
    hist = cp.zeros((nt, N), DTYPE) if store else None
    grid = _grid(N)
    for n in range(1, nt):
        _fwd_step(grid, (_BLK,), (p_cur, p_prev, S, p_new, nz, ny, nx, inv_h2))
        p_new[src_lin] += Sw * wavelet[n - 1]
        p_prev, p_cur, p_new = p_cur, p_new, p_prev
        if store:
            hist[n] = p_cur
        rec[n] = p_cur[rec_lin]
    return rec, hist, S


def _mg3d(mflat, dims, h, dt, nt, tx_lin, rec_lin, wavelet, dobs):
    nz, ny, nx = dims
    N = mflat.size
    inv_h2 = np.float32(1.0 / (h * h))
    grid = _grid(N)
    g = cp.zeros(N, DTYPE)
    zeros = cp.zeros(N, DTYPE)
    J = 0.0
    for i in range(tx_lin.size):
        rec, U, S = _forward3d(mflat, dims, h, dt, nt, int(tx_lin[i]), wavelet, rec_lin, store=True)
        res = rec - dobs[i]
        J += 0.5 * float(cp.sum(res * res))
        lam_p1 = cp.zeros(N, DTYPE); lam_p2 = cp.zeros(N, DTYPE); lam_k = cp.zeros(N, DTYPE)
        for k in range(nt - 1, 0, -1):
            _adj_step(grid, (_BLK,), (lam_p1, lam_p2, S, lam_k, nz, ny, nx, inv_h2))
            lam_k[rec_lin] -= res[k]
            uk2 = U[k - 2] if k >= 2 else zeros
            _grad_acc(lam_k, U[k], U[k - 1], uk2, g)
            lam_p2, lam_p1, lam_k = lam_p1, lam_k, lam_p2
    g /= mflat
    return J, g


# ---------------------------------------------------------------------------
# 2D sliced path
# ---------------------------------------------------------------------------
def _forward2d(m, h, dt, nt, src_idx, wavelet, rec_ax, store):
    inv_h2 = np.float32(1.0 / (h * h))
    S = (np.float32(dt * dt)) / m
    p_prev = cp.zeros(m.shape, DTYPE); p_cur = cp.zeros(m.shape, DTYPE); lap = cp.zeros(m.shape, DTYPE)
    rec = cp.zeros((nt, rec_ax[0].size), DTYPE)
    hist = cp.zeros((nt,) + m.shape, DTYPE) if store else None
    for n in range(1, nt):
        _laplacian(p_cur, inv_h2, lap)
        lap[src_idx] += wavelet[n - 1]
        p_new = 2.0 * p_cur - p_prev + S * lap
        p_prev, p_cur = p_cur, p_new
        if store:
            hist[n] = p_cur
        rec[n] = p_cur[rec_ax]
    return rec, hist, S


# ---------------------------------------------------------------------------
# public API (signature-compatible with ringfwi.fwi)
# ---------------------------------------------------------------------------
def forward_fmc(m, geom, wavelet, dt, h, nt, sponge=None, src_list=None):
    m = cp.asarray(m, DTYPE)
    wav = cp.asarray(wavelet, DTYPE)
    if src_list is None:
        src_list = list(range(geom.n_elements))
    if m.ndim == 3:
        dims = m.shape
        mflat = m.ravel()
        rec_lin = _lin_indices(geom, dims)
        data = cp.zeros((len(src_list), nt, geom.n_elements), DTYPE)
        for i, s in enumerate(src_list):
            src_lin = int(rec_lin[s])
            rec, _, _ = _forward3d(mflat, dims, h, dt, nt, src_lin, wav, rec_lin, store=False)
            data[i] = rec
        return cp.asnumpy(data)
    rec_ax = _axes(geom, 2)
    data = cp.zeros((len(src_list), nt, geom.n_elements), DTYPE)
    for i, s in enumerate(src_list):
        rec, _, _ = _forward2d(m, h, dt, nt, geom.element_index(s), wav, rec_ax, store=False)
        data[i] = rec
    return cp.asnumpy(data)


def misfit_and_gradient(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
                        misfit_type="l2"):
    if misfit_type != "l2":
        raise NotImplementedError("GPU backend implements the L2 misfit only")
    m = cp.asarray(m, DTYPE)
    wav = cp.asarray(wavelet, DTYPE)
    dobs = cp.asarray(dobs, DTYPE)
    if src_list is None:
        src_list = list(range(geom.n_elements))

    if m.ndim == 3:
        dims = m.shape
        rec_lin = _lin_indices(geom, dims)
        tx_lin = cp.asarray([int(rec_lin[s]) for s in src_list])
        J, gflat = _mg3d(m.ravel(), dims, h, dt, nt, tx_lin, rec_lin, wav, dobs)
        return J, np.asarray(cp.asnumpy(gflat.reshape(dims)), dtype=np.float64)

    # 2D sliced path.
    rec_ax = _axes(geom, 2)
    inv_h2 = np.float32(1.0 / (h * h))
    g = cp.zeros(m.shape, DTYPE); lap = cp.zeros(m.shape, DTYPE)
    J = 0.0
    for i, s in enumerate(src_list):
        dsyn, U, S = _forward2d(m, h, dt, nt, geom.element_index(s), wav, rec_ax, store=True)
        res = dsyn - dobs[i]
        J += 0.5 * float(cp.sum(res * res))
        lam_p1 = cp.zeros(m.shape, DTYPE); lam_p2 = cp.zeros(m.shape, DTYPE)
        for k in range(nt - 1, 0, -1):
            _laplacian(S * lam_p1, inv_h2, lap)
            lam_k = 2.0 * lam_p1 - lam_p2 + lap
            lam_k[rec_ax] -= res[k]
            d2p = U[k] - 2.0 * U[k - 1] + (U[k - 2] if k >= 2 else 0.0)
            g += lam_k * d2p
            lam_p2, lam_p1 = lam_p1, lam_k
    g /= m
    return J, np.asarray(cp.asnumpy(g), dtype=np.float64)


def misfit(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
           misfit_type="l2"):
    if misfit_type != "l2":
        raise NotImplementedError("GPU backend implements the L2 misfit only")
    dsyn = forward_fmc(m, geom, wavelet, dt, h, nt, src_list=src_list)
    r = dsyn - dobs
    return 0.5 * float(np.sum(r * r))
