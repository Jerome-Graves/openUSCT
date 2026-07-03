"""Voxel-wise c-axis orientation-field FWI in 3D (unknown grain geometry).

Every voxel of the specimen carries its own crystal orientation (colatitude,
azimuth) -- your "each voxel is a function of velocity at every angle" idea
taken literally, with no grain labels anywhere. Grains are not an input: they
emerge as regions where the recovered axis field is constant.

Machinery:
* :func:`axis_stiffness_and_derivs` -- per-voxel rotated 21-component
  stiffness plus analytic d/d(colatitude), d/d(azimuth) via the Bond 6x6
  transformation (exact, vectorised over voxels).
* :func:`misfit_and_gradient_axes` -- chains the exact 21-map adjoint of
  :mod:`ringfwi.elastic3d` through the rotation, giving the exact per-voxel
  orientation gradient.
* :func:`block_partition` -- coarse regular blocks over the specimen mask, so
  the globally-searchable COF machinery (:mod:`ringfwi.cof`) can initialise
  the field without knowing the true geometry.
* :func:`smooth_axes` -- orientation-tensor (Q = a a^T) smoothing, the
  c/-c-safe regulariser that promotes grain-like piecewise-constant fields.
"""

from __future__ import annotations

import numpy as np

from . import anisotropy as _an
from . import elastic3d as _e3d

_VOIGT3 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
KEYS21 = tuple(f"C{i}{j}" for i in range(1, 7) for j in range(i, 7))


# ---------------------------------------------------------------------------
# Rotation of the stiffness tensor and its analytic angle derivatives.
# ---------------------------------------------------------------------------

def rotations(colat, azim):
    """R = Rz(azim) Ry(colat) per voxel, plus dR/dcolat and dR/dazim.

    ``colat``/``azim`` are flat (P,) arrays; returns three (P, 3, 3) arrays.
    R maps the crystal frame (c-axis along z) into the lab frame:
    R zhat = (sin t cos p, sin t sin p, cos t) -- the c-axis of
    :func:`ringfwi.cof.axes_from_params`.
    """
    t = np.asarray(colat, float).ravel()
    p = np.asarray(azim, float).ravel()
    ct, st = np.cos(t), np.sin(t)
    cp, sp = np.cos(p), np.sin(p)
    P = t.size
    Ry = np.zeros((P, 3, 3)); dRy = np.zeros((P, 3, 3))
    Ry[:, 0, 0] = ct;  Ry[:, 0, 2] = st
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -st; Ry[:, 2, 2] = ct
    dRy[:, 0, 0] = -st; dRy[:, 0, 2] = ct
    dRy[:, 2, 0] = -ct; dRy[:, 2, 2] = -st
    Rz = np.zeros((P, 3, 3)); dRz = np.zeros((P, 3, 3))
    Rz[:, 0, 0] = cp; Rz[:, 0, 1] = -sp
    Rz[:, 1, 0] = sp; Rz[:, 1, 1] = cp
    Rz[:, 2, 2] = 1.0
    dRz[:, 0, 0] = -sp; dRz[:, 0, 1] = -cp
    dRz[:, 1, 0] = cp;  dRz[:, 1, 1] = -sp
    R = Rz @ Ry
    dR_t = Rz @ dRy
    dR_p = dRz @ Ry
    return R, dR_t, dR_p


def _bond2(A, B):
    """Bilinear Bond form: (P,3,3),(P,3,3) -> (P,6,6).

    ``_bond2(R, R)`` is the Bond stress-transformation matrix M(R) with
    C' = M C M^T for engineering-strain Voigt stiffness; the bilinearity gives
    dM = _bond2(dR, R) + _bond2(R, dR).
    """
    P = A.shape[0]
    M = np.zeros((P, 6, 6))
    for a, (i, j) in enumerate(_VOIGT3):
        for b, (k, l) in enumerate(_VOIGT3):
            M[:, a, b] = A[:, i, k] * B[:, j, l]
            if k != l:
                M[:, a, b] += A[:, i, l] * B[:, j, k]
    return M


def axis_stiffness_and_derivs(colat, azim, base6):
    """Per-voxel rotated stiffness and its exact angle derivatives.

    Returns (C, dC_t, dC_p), each (P, 6, 6): the stiffness of ``base6``
    (crystal frame, c along z) rotated so the c-axis points along
    (colat, azim), and its derivatives with respect to the two angles.
    """
    base6 = np.asarray(base6, float)
    R, dR_t, dR_p = rotations(colat, azim)
    M = _bond2(R, R)
    dM_t = _bond2(dR_t, R) + _bond2(R, dR_t)
    dM_p = _bond2(dR_p, R) + _bond2(R, dR_p)
    MC = M @ base6
    C = MC @ np.swapaxes(M, 1, 2)
    dC_t = (dM_t @ base6) @ np.swapaxes(M, 1, 2) + MC @ np.swapaxes(dM_t, 1, 2)
    dC_p = (dM_p @ base6) @ np.swapaxes(M, 1, 2) + MC @ np.swapaxes(dM_p, 1, 2)
    return C, dC_t, dC_p


def build_maps(colat_map, azim_map, mask, base6, Cbg, rho_bg, rho_mat):
    """Full-grid 21 stiffness maps from a voxel axis field.

    Voxels inside ``mask`` get the per-voxel rotated crystal stiffness;
    everywhere else keeps the background maps ``Cbg`` (couplant). Returns
    (Cmaps, rho).
    """
    C, _, _ = axis_stiffness_and_derivs(colat_map[mask], azim_map[mask], base6)
    Cmaps = {}
    for key in KEYS21:
        i, j = int(key[1]) - 1, int(key[2]) - 1
        m = np.array(Cbg[key], float, copy=True)
        m[mask] = C[:, i, j]
        Cmaps[key] = m
    rho = np.full(mask.shape, float(rho_bg))
    rho[mask] = float(rho_mat)
    return Cmaps, rho


def misfit_and_gradient_axes(colat_map, azim_map, mask, base6, Cbg, rho_bg,
                             rho_mat, h, dt, nt, src_pts_list, wavelet,
                             rec_groups, dobs, trace_weights=None):
    """Total misfit and exact per-voxel orientation gradient, all sources.

    ``src_pts_list`` is a list (one per transmit) of element footprints;
    ``dobs`` is (n_tx, nt, n_rx); ``trace_weights`` broadcasts against a
    single shot (nt, n_rx) or is a list per transmit. Returns
    (J, g_colat, g_azim) with the gradients full-grid (zero off the mask).
    """
    Cmaps, rho = build_maps(colat_map, azim_map, mask, base6, Cbg,
                            rho_bg, rho_mat)
    J = 0.0
    g21 = {k: np.zeros(mask.shape) for k in KEYS21}
    for i_tx, src_pts in enumerate(src_pts_list):
        tw = (trace_weights[i_tx] if isinstance(trace_weights, (list, tuple))
              else trace_weights)
        Js, g = _e3d.misfit_and_gradient(
            Cmaps, rho, h, dt, nt, None, wavelet, None, dobs[i_tx],
            src_pts=src_pts, rec_groups=rec_groups, trace_weights=tw)
        J += Js
        for k in KEYS21:
            g21[k] += g[k]

    _, dC_t, dC_p = axis_stiffness_and_derivs(colat_map[mask],
                                              azim_map[mask], base6)
    g_t = np.zeros(mask.shape); g_p = np.zeros(mask.shape)
    gt = np.zeros(int(mask.sum())); gp = np.zeros_like(gt)
    for key in KEYS21:
        i, j = int(key[1]) - 1, int(key[2]) - 1
        gk = g21[key][mask]
        gt += gk * dC_t[:, i, j]
        gp += gk * dC_p[:, i, j]
    g_t[mask] = gt; g_p[mask] = gp
    return J, g_t, g_p


# ---------------------------------------------------------------------------
# Coarse initialisation and orientation-tensor regularisation.
# ---------------------------------------------------------------------------

def block_partition(mask, n_div):
    """Regular ``n_div``^3 block labels over the mask's bounding box.

    Returns (labels, n_blocks) with labels = -1 outside the mask and blocks
    renumbered densely (empty blocks dropped). These are pseudo-grains for
    the global coarse search -- they need not match any true geometry.
    """
    labels = np.full(mask.shape, -1, dtype=int)
    zz, yy, xx = np.nonzero(mask)
    edges = []
    for lo, hi in ((zz.min(), zz.max() + 1), (yy.min(), yy.max() + 1),
                   (xx.min(), xx.max() + 1)):
        edges.append(np.linspace(lo, hi, n_div + 1))
    bz = np.clip(np.searchsorted(edges[0], zz, side="right") - 1, 0, n_div - 1)
    by = np.clip(np.searchsorted(edges[1], yy, side="right") - 1, 0, n_div - 1)
    bx = np.clip(np.searchsorted(edges[2], xx, side="right") - 1, 0, n_div - 1)
    raw = (bz * n_div + by) * n_div + bx
    uniq, dense = np.unique(raw, return_inverse=True)
    labels[zz, yy, xx] = dense
    return labels, len(uniq)


def axes_to_field(labels, axes, mask):
    """Per-block axes -> per-voxel (colat, azim) maps over ``mask``."""
    from .cof import params_from_axes
    par = params_from_axes(axes)
    colat = np.zeros(mask.shape); azim = np.zeros(mask.shape)
    for k in range(len(axes)):
        m = labels == k
        colat[m] = par[k, 0]; azim[m] = par[k, 1]
    return colat, azim


def field_to_axes(colat_map, azim_map):
    """(colat, azim) maps -> unit axis vectors, shape (nz, ny, nx, 3)."""
    st, ct = np.sin(colat_map), np.cos(colat_map)
    return np.stack([st * np.cos(azim_map), st * np.sin(azim_map), ct],
                    axis=-1)


def smooth_axes(colat_map, azim_map, mask, sigma):
    """Orientation-tensor smoothing of the axis field (c/-c safe).

    Averages Q = a a^T with a Gaussian (restricted to the mask) and returns
    the principal eigenvector's angles -- the regulariser that respects the
    axis topology (no angle-wrap seams) and promotes grain-like
    piecewise-constant fields.
    """
    from scipy.ndimage import gaussian_filter
    a = field_to_axes(colat_map, azim_map)
    w = gaussian_filter(mask.astype(float), sigma) + 1e-12
    Q = np.zeros(mask.shape + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            q = gaussian_filter(a[..., i] * a[..., j] * mask, sigma) / w
            Q[..., i, j] = q
            Q[..., j, i] = q
    vals, vecs = np.linalg.eigh(Q[mask])
    v = vecs[..., -1]                                   # principal axis
    v = np.where(v[:, 2:3] < 0, -v, v)                  # upper hemisphere
    colat = colat_map.copy(); azim = azim_map.copy()
    colat[mask] = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
    azim[mask] = np.arctan2(v[:, 1], v[:, 0])
    return colat, azim


def gradient_step(colat_map, azim_map, mask, g_t, g_p, step_rad=0.1,
                  smooth_sigma=0.0):
    """Normalised steepest-descent update of the axis field.

    ``smooth_sigma`` > 0 Gaussian-smooths the GRADIENT (mask-normalised)
    before the step -- the regularisation that worked for the 2D theta-field
    FWI. Smoothing the field itself after each step fights the data term
    (measured: the misfit rises); keep :func:`smooth_axes` for one-off field
    polishing only. Both angle maps move under one shared normalisation (so
    azimuth, whose sensitivity vanishes near the poles, is not artificially
    inflated).
    """
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        w = gaussian_filter(mask.astype(float), smooth_sigma) + 1e-12
        g_t = gaussian_filter(g_t * mask, smooth_sigma) / w * mask
        g_p = gaussian_filter(g_p * mask, smooth_sigma) / w * mask
    gmax = max(np.max(np.abs(g_t[mask])), np.max(np.abs(g_p[mask]))) + 1e-30
    colat = colat_map - step_rad * (g_t / gmax) * mask
    azim = azim_map - step_rad * (g_p / gmax) * mask
    return colat, azim


def axis_error_map(colat_map, azim_map, true_axes_map, mask):
    """Per-voxel angular error (degrees) against a true axis field."""
    a = field_to_axes(colat_map, azim_map)
    dot = np.abs(np.sum(a * true_axes_map, axis=-1))
    err = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
    return np.where(mask, err, 0.0)
