"""Grain-parameterised crystal-orientation-fabric (COF) inversion.

Recovers per-grain c-axis orientations from elastic full-matrix-capture data
with known grain geometry and material: the unknowns are two angles per grain
(colatitude, azimuth), so the Jacobian of the residual vector is cheap to form
by finite differences and Gauss-Newton / Levenberg-Marquardt "Jacobian
iterations" converge quadratically near the optimum — the full-waveform
upgrade of the reduced-forward-model Jacobian iteration used for travel-time
COF estimation.

Typical use: a coarse per-grain hemisphere search for initialisation (global),
then :func:`jacobian_iterations` for fast, precise convergence.
"""

from __future__ import annotations

import numpy as np

from . import anisotropy as _an
from . import elastic3d as _e3d


def axes_from_params(params):
    """(G, 2) [colatitude, azimuth] (radians) -> (G, 3) unit c-axis vectors."""
    p = np.asarray(params, float).reshape(-1, 2)
    th, ph = p[:, 0], p[:, 1]
    return np.column_stack([np.sin(th) * np.cos(ph),
                            np.sin(th) * np.sin(ph),
                            np.cos(th)])


def params_from_axes(axes):
    """(G, 3) unit axes -> (G, 2) [colatitude, azimuth]; upper hemisphere."""
    a = np.asarray(axes, float).reshape(-1, 3)
    a = np.where(a[:, 2:3] < 0, -a, a)                  # c and -c are equivalent
    th = np.arccos(np.clip(a[:, 2], -1.0, 1.0))
    ph = np.arctan2(a[:, 1], a[:, 0])
    return np.column_stack([th, ph])


def fmc(labels, axes, ring, h, dt, nt, wavelet, src_list, material=None,
        device="auto"):
    """Elastic full-matrix capture for a polycrystal with the given axes.

    ``device="auto"`` runs on the GPU (CuPy, float32) for grids large enough
    to benefit; pass "cpu" for the float64 reference.
    """
    Cm, rho = _an.polycrystal_stiffness_3d(labels, axes, material=material)
    data = np.zeros((len(src_list), nt, ring.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = _e3d.forward(Cm, rho, h, dt, nt, ring.element_index(s),
                              wavelet, ring.idx, source="explosive",
                              record="pressure", device=device)
        data[i] = rec
    return data


def make_residual(labels, ring, h, dt, nt, wavelet, src_list, dobs,
                  material=None, filter_fn=None, device="auto"):
    """Per-trace-normalised residual vector r(params) and its half-SSQ misfit.

    Each source-receiver trace is normalised by the observed trace energy and
    the transmitter's own trace is excluded (its near-source blast carries no
    orientation information and would otherwise dominate).

    ``filter_fn`` is applied to the synthetic data before differencing (pass
    the same RX front-end filter the observed data went through; the chain is
    linear so filtering both sides is consistent).

    Returns ``residual(params) -> r`` with J(params) = 0.5 * r.r matching the
    misfit used by the search-based COF inversion.
    """
    tr_norm = np.sqrt(np.sum(dobs ** 2, axis=1)) + 1e-30    # (n_tx, n_rx)
    w = np.ones_like(tr_norm)
    for i, s in enumerate(src_list):
        w[i, s] = 0.0

    def residual(params):
        d = fmc(labels, axes_from_params(params), ring, h, dt, nt, wavelet,
                src_list, material=material, device=device)
        if filter_fn is not None:
            d = filter_fn(d)
        r = (d - dobs) * (w / tr_norm)[:, None, :]
        return r.ravel()

    return residual


def jacobian_iterations(residual, params0, n_iter=6, fd_step=np.radians(2.0),
                        lam0=1e-2, tol=1e-12, verbose=False):
    """Levenberg-Marquardt Gauss-Newton on a finite-difference Jacobian.

    ``residual(params_flat) -> r``; the Jacobian is formed column-by-column by
    forward differences with angular step ``fd_step``. The damping ``lam`` is
    adapted per iteration (decrease on success, increase and retry on failure).

    Returns (params, history) with ``history`` the half-SSQ misfit per accepted
    iteration (first entry = starting misfit).
    """
    p = np.asarray(params0, float).ravel().copy()
    n_par = p.size
    r = residual(p)
    J_val = 0.5 * float(r @ r)
    history = [J_val]
    lam = lam0

    for it in range(n_iter):
        # finite-difference Jacobian (one forward evaluation per parameter)
        Jac = np.empty((r.size, n_par))
        for k in range(n_par):
            pk = p.copy()
            pk[k] += fd_step
            Jac[:, k] = (residual(pk) - r) / fd_step

        g = Jac.T @ r
        H = Jac.T @ Jac
        accepted = False
        for _try in range(6):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-30), -g)
            p_try = p + step
            r_try = residual(p_try)
            J_try = 0.5 * float(r_try @ r_try)
            if J_try < J_val:
                p, r, J_val = p_try, r_try, J_try
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 5.0
        history.append(J_val)
        if verbose:
            print(f"  LM iter {it}: J={J_val:.6e} lam={lam:.1e} "
                  f"{'accepted' if accepted else 'stalled'}", flush=True)
        if not accepted or J_val < tol:
            break

    return p.reshape(-1, 2), history


def axis_error_deg(a, b):
    """Angle between two c-axes (degrees), respecting the c/-c symmetry."""
    return float(np.degrees(np.arccos(np.clip(
        abs(float(np.dot(np.asarray(a, float), np.asarray(b, float)))), 0.0, 1.0))))
