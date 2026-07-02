"""Multi-parameter FWI: sound speed and attenuation.

Extends the acoustic model with a viscous loss term so that both the
squared slowness ``m = 1/c^2`` and an attenuation field ``a`` can be
reconstructed. The forward model is the damped wave equation

    m d2p/dt2 + a dp/dt = laplacian(p) + f

discretised with a time-symmetric leapfrog (the a-term centred as
(p^n - p^{n-2}) / 2dt), which keeps the scheme self-adjoint. The exact discrete
adjoint yields gradients for *both* parameters:

    g_m(x) = (1/dt^2) sum_n lam^n (p^n - 2 p^{n-1} + p^{n-2})
    g_a(x) = (1/2dt)  sum_n lam^n (p^n - p^{n-2})

Both are verified against finite differences in tests/test_gradient_attenuation.py.
"""

from __future__ import annotations

import numpy as np

from .solver import _laplacian


def _forward(m, a, h, dt, nt, sources, rec_idx=None, store=False):
    """Damped leapfrog. Returns (rec, hist)."""
    shape = m.shape
    inv_h2 = 1.0 / (h * h)
    alpha = m / (dt * dt)
    beta = a / (2.0 * dt)
    invA = 1.0 / (alpha + beta)

    p_nm2 = np.zeros(shape)   # p^{n-2}
    p_nm1 = np.zeros(shape)   # p^{n-1}
    lap = np.zeros(shape)

    rec = None if rec_idx is None else np.zeros((nt, len(rec_idx)))
    hist = None if not store else np.zeros((nt,) + shape)

    for n in range(1, nt):
        _laplacian(p_nm1, inv_h2, lap)
        for idx, series in sources:
            lap[idx] += series[n - 1]
        p_n = invA * (2.0 * alpha * p_nm1 + (beta - alpha) * p_nm2 + lap)
        p_nm2, p_nm1 = p_nm1, p_n
        if hist is not None:
            hist[n] = p_n
        if rec is not None:
            for j, idx in enumerate(rec_idx):
                rec[n, j] = p_n[idx]
    return rec, hist


def forward_fmc(m, a, geom, wavelet, dt, h, nt, src_list=None):
    """Full-matrix-capture forward with attenuation."""
    src_list = list(range(geom.n_elements)) if src_list is None else list(src_list)
    rec_idx = geom.idx
    data = np.zeros((len(src_list), nt, geom.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = _forward(m, a, h, dt, nt, [(geom.element_index(s), wavelet)], rec_idx)
        data[i] = rec
    return data


def misfit(m, a, geom, wavelet, dt, h, nt, dobs, src_list=None):
    """Least-squares waveform misfit (forward only)."""
    dsyn = forward_fmc(m, a, geom, wavelet, dt, h, nt, src_list)
    r = dsyn - dobs
    return 0.5 * float(np.sum(r * r))


def misfit_and_gradients(m, a, geom, wavelet, dt, h, nt, dobs, src_list=None):
    """Misfit and adjoint-state gradients w.r.t. both m and a.

    Returns (J, g_m, g_a).
    """
    src_list = list(range(geom.n_elements)) if src_list is None else list(src_list)
    rec_idx = geom.idx
    shape = m.shape
    inv_h2 = 1.0 / (h * h)
    alpha = m / (dt * dt)
    beta = a / (2.0 * dt)
    invA = 1.0 / (alpha + beta)

    J = 0.0
    g_alpha = np.zeros(shape)
    g_beta = np.zeros(shape)
    lap = np.zeros(shape)

    for i, s in enumerate(src_list):
        dsyn, U = _forward(m, a, h, dt, nt, [(geom.element_index(s), wavelet)],
                           rec_idx, store=True)
        res = dsyn - dobs[i]
        J += 0.5 * float(np.sum(res * res))

        lam_p1 = np.zeros(shape)   # lam^{k+1}
        lam_p2 = np.zeros(shape)   # lam^{k+2}
        for k in range(nt - 1, 0, -1):
            _laplacian(lam_p1, inv_h2, lap)
            lam_k = 2.0 * alpha * lam_p1 + (beta - alpha) * lam_p2 + lap
            for j, idx in enumerate(rec_idx):
                lam_k[idx] -= res[k, j]
            lam_k = lam_k * invA

            Uk = U[k]
            Uk1 = U[k - 1]
            Uk2 = U[k - 2] if k >= 2 else 0.0
            g_alpha += lam_k * (Uk - 2.0 * Uk1 + Uk2)
            g_beta += lam_k * (Uk - Uk2)

            lam_p2, lam_p1 = lam_p1, lam_k

    g_m = g_alpha / (dt * dt)
    g_a = g_beta / (2.0 * dt)
    return J, g_m, g_a


def invert(m0, a0, geom, wavelet, dt, h, nt, dobs, src_list=None, n_iter=12,
           step_m=0.03, step_a=0.06, update_mask=None, m_bounds=None, a_bounds=None,
           smooth=1.0, verbose=False):
    """Joint steepest-descent FWI for sound speed and attenuation.

    The two parameters have very different scales, so each takes its own
    normalised step. Returns (m, a, history).
    """
    from scipy.ndimage import gaussian_filter

    m = m0.copy()
    a = a0.copy()
    m_start, a_start = m0.copy(), a0.copy()
    history = []
    for it in range(n_iter):
        J, gm, ga = misfit_and_gradients(m, a, geom, wavelet, dt, h, nt, dobs, src_list)
        history.append(J)
        if verbose:
            print(f"  iter {it:2d}  misfit = {J:.6e}")
        for g in (gm, ga):
            if update_mask is not None:
                g *= update_mask
        if smooth:
            gm = gaussian_filter(gm, smooth); ga = gaussian_filter(ga, smooth)
            if update_mask is not None:
                gm *= update_mask; ga *= update_mask
        gmx, gax = float(np.max(np.abs(gm))), float(np.max(np.abs(ga)))
        if gmx > 0:
            m = m - step_m * float(np.mean(m)) * gm / gmx
        if gax > 0:
            a = a - step_a * float(np.mean(a)) * ga / gax
        if m_bounds is not None:
            cm = np.clip(m, m_bounds[0], m_bounds[1])
            m = np.where(update_mask > 0, cm, m) if update_mask is not None else cm
        if a_bounds is not None:
            ca = np.clip(a, a_bounds[0], a_bounds[1])
            a = np.where(update_mask > 0, ca, a) if update_mask is not None else ca
    history.append(misfit(m, a, geom, wavelet, dt, h, nt, dobs, src_list))
    return m, a, history
