"""Voronoi-seed COF inversion: unknown grain geometry as a low-dimensional
parametric unknown.

The literature answer to the voxel-field null space (Bodin & Sambridge 2009,
transdimensional Voronoi tomography; Kadu et al. 2017, parametric level-set
FWI): when the target is piecewise-constant with unknown geometry, invert the
GEOMETRY jointly with the properties at low dimension instead of thousands of
voxels. Here the model is G Voronoi seeds with a c-axis each -- five
parameters per grain: [x, y, z, colatitude, azimuth] -- and the verified
Levenberg-Marquardt machinery of :mod:`ringfwi.cof` closes the loop. The
known-geometry COF inversion is this method with the seeds frozen.

For differentiability the inversion uses a SOFT Voronoi assignment (softmax
of negative squared distance, temperature ``tau_voxels`` grid cells): seed
motion then changes the misfit smoothly instead of voxel-by-voxel staircase
jumps, so finite-difference Jacobians are well behaved. Reported results use
the hard (argmin) labels.
"""

from __future__ import annotations

import numpy as np

from . import anisotropy as _an
from . import axisfield as _af
from . import elastic3d as _e3d

_KEYS21 = _af.KEYS21


def grid_coords(shape, h):
    """Voxel-centre physical coordinates, columns (x, y, z), metres."""
    nz, ny, nx = shape
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float) * h
    return x, y, z


def soft_weights(seeds, mask, h, tau_voxels=1.5):
    """(P, G) soft Voronoi weights over the mask voxels.

    softmax(-d^2 / (2 (tau h)^2)) -- smooth in the seed positions, and the
    hard argmin assignment in the limit tau -> 0.
    """
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    x, y, z = grid_coords(mask.shape, h)
    P = np.stack([x[mask], y[mask], z[mask]], axis=1)      # (P, 3)
    d2 = ((P[:, None, :] - seeds[None]) ** 2).sum(-1)      # (P, G)
    s2 = 2.0 * (tau_voxels * h) ** 2
    a = -d2 / s2
    a -= a.max(axis=1, keepdims=True)
    w = np.exp(a)
    w /= w.sum(axis=1, keepdims=True)
    return w


def hard_labels(seeds, mask, h):
    """Nearest-seed labels over the mask (-1 outside), like the phantom."""
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    x, y, z = grid_coords(mask.shape, h)
    P = np.stack([x[mask], y[mask], z[mask]], axis=1)
    d2 = ((P[:, None, :] - seeds[None]) ** 2).sum(-1)
    lab = np.full(mask.shape, -1, dtype=int)
    lab[mask] = np.argmin(d2, axis=1)
    return lab


def params_pack(seeds, axes):
    """(G,3) seeds + (G,3) axes -> flat (G*5,) [x y z colat azim] params."""
    from .cof import params_from_axes
    ang = params_from_axes(axes)
    return np.concatenate([np.asarray(seeds, float),
                           ang], axis=1).ravel()


def params_unpack(params):
    """Flat params -> ((G,3) seeds, (G,3) axes)."""
    from .cof import axes_from_params
    p = np.asarray(params, float).reshape(-1, 5)
    return p[:, :3].copy(), axes_from_params(p[:, 3:5])


def blended_maps(seeds, axes, mask, h, base6, Cbg, rho_bg, rho_mat,
                 tau_voxels=1.5):
    """Full-grid 21 stiffness maps from soft-Voronoi-blended grain tensors."""
    from .cof import params_from_axes
    ang = params_from_axes(axes)
    C6g, _, _ = _af.axis_stiffness_and_derivs(ang[:, 0], ang[:, 1], base6)
    w = soft_weights(seeds, mask, h, tau_voxels)           # (P, G)
    Cmaps = {}
    for key in _KEYS21:
        i, j = int(key[1]) - 1, int(key[2]) - 1
        m = np.array(Cbg[key], float, copy=True)
        m[mask] = w @ C6g[:, i, j]
        Cmaps[key] = m
    rho = np.full(mask.shape, float(rho_bg))
    rho[mask] = float(rho_mat)
    return Cmaps, rho


def make_residual_seeds(mask, ring, h, dt, nt, wavelet, src_list, dobs,
                        material=None, filter_fn=None, tau_voxels=1.5,
                        device="auto"):
    """Per-trace-normalised residual r(params) over [seeds, angles] per grain.

    Same misfit convention as :func:`ringfwi.cof.make_residual` (observed
    trace-energy normalisation, transmitter's own trace excluded), so J values
    are directly comparable with the known-geometry COF inversion.
    """
    mat = _an.ICE_MATERIAL if material is None else material
    base6 = _an.ti_stiffness_6(**mat)
    K = 1000.0 * 1480.0 ** 2
    Cbg = {k: np.full(mask.shape,
                      K if (int(k[1]) <= 3 and int(k[2]) <= 3) else 0.0)
           for k in _KEYS21}
    tr_norm = np.sqrt(np.sum(dobs ** 2, axis=1)) + 1e-30
    w = np.ones_like(tr_norm)
    for i, s in enumerate(src_list):
        w[i, s] = 0.0

    def residual(params):
        seeds, axes = params_unpack(params)
        Cmaps, rho = blended_maps(seeds, axes, mask, h, base6, Cbg,
                                  1000.0, mat["rho"], tau_voxels)
        src_pts_list = [[(ring.element_index(s), 1.0)] for s in src_list]
        d = _e3d.forward_batch(Cmaps, rho, h, dt, nt, src_pts_list, wavelet,
                               rec_idx=ring.idx, device=device)
        if filter_fn is not None:
            d = filter_fn(d)
        r = (d - dobs) * (w / tr_norm)[:, None, :]
        return r.ravel()

    return residual


def kmeans_seeds(mask, h, n_grains, seed=0, iters=25):
    """Geometric k-means of the specimen voxel coordinates -> (G, 3) seeds.

    A neutral, geometry-free initial seed layout (roughly equal-volume cells
    spread through the specimen).
    """
    x, y, z = grid_coords(mask.shape, h)
    P = np.stack([x[mask], y[mask], z[mask]], axis=1)
    rng = np.random.default_rng(seed)
    C = P[rng.choice(len(P), size=n_grains, replace=False)]
    for _ in range(iters):
        idx = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for k in range(n_grains):
            m = idx == k
            if m.any():
                C[k] = P[m].mean(0)
    return C


def fd_steps(n_grains, h, seed_step_voxels=1.0, angle_step_deg=2.0):
    """Per-parameter FD step vector matching the [x y z colat azim] layout."""
    one = np.array([seed_step_voxels * h] * 3
                   + [np.radians(angle_step_deg)] * 2)
    return np.tile(one, n_grains)
