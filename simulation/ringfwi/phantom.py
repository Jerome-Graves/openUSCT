"""Synthetic velocity phantoms for a cylindrical specimen.

Models are returned as squared-slowness fields (m = 1 / c**2) because that is
the parameter the solver and the FWI gradient use directly. Helper conversions
between velocity and squared slowness are provided.
"""

from __future__ import annotations

import numpy as np


def velocity_to_m(c):
    return 1.0 / (c * c)


def m_to_velocity(m):
    return 1.0 / np.sqrt(m)


def coupling_background(shape, c_specimen, c_couplant, radius_m, h):
    """Circular specimen of velocity ``c_specimen`` in a ``c_couplant`` bath.

    Returns a velocity field. The specimen disc is centred in the domain; the
    surrounding couplant (for example water) fills the corners so the ring array
    always sees a continuous medium.
    """
    n = shape[0]
    centre = (n - 1) * h / 2.0
    y, x = np.mgrid[0:n, 0:n] * h
    r = np.hypot(x - centre, y - centre)
    c = np.where(r <= radius_m, c_specimen, c_couplant)
    return c.astype(float)


def add_inclusion(c, centre_frac, radius_m, c_inclusion, h):
    """Add a circular velocity inclusion (a flaw / anomaly) to a velocity field.

    ``centre_frac`` is (fx, fy) in [0, 1] domain fractions.
    """
    n = c.shape[0]
    domain = (n - 1) * h
    cx, cy = centre_frac[0] * domain, centre_frac[1] * domain
    y, x = np.mgrid[0:n, 0:n] * h
    r = np.hypot(x - cx, y - cy)
    out = c.copy()
    out[r <= radius_m] = c_inclusion
    return out


def smooth(field, sigma_cells):
    """Gaussian smoothing (used to build a starting model for FWI)."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma_cells)


def voronoi_polycrystal(shape, n_grains, radius_m, h, rng=None, relax=1):
    """Voronoi polycrystal filling a circular specimen (2D).

    Models a polycrystalline solid such as glacial ice: the specimen disc is
    tessellated into ``n_grains`` Voronoi cells (grains), each a single crystal
    with its own crystallographic orientation. Grain shapes come from random
    seed points assigned by nearest-seed distance (which *is* the Voronoi
    tessellation, evaluated on the grid); ``relax`` Lloyd iterations move each
    seed to its cell centroid for more equiaxed, physically plausible grains.

    Returns
    -------
    labels : (n, n) int ndarray
        Grain index per cell, -1 outside the specimen disc.
    angles : (n_grains,) ndarray
        One c-axis angle per grain, uniform on [0, pi).
    theta_map : (n, n) ndarray
        Per-cell c-axis angle (0 outside the specimen).
    """
    if len(shape) != 2:
        raise ValueError("voronoi_polycrystal is 2D (3D polycrystal is roadmap)")
    if rng is None:
        rng = np.random.default_rng()
    n = shape[0]
    c = (n - 1) * h / 2.0
    y, x = np.mgrid[0:n, 0:n].astype(float) * h
    inside = np.hypot(x - c, y - c) <= radius_m

    # Seeds uniform in the disc (rejection sampling).
    seeds = []
    while len(seeds) < n_grains:
        p = rng.uniform(-radius_m, radius_m, size=2)
        if np.hypot(*p) <= radius_m:
            seeds.append(p + c)
    seeds = np.asarray(seeds)

    def _labels(s):
        d2 = (x[..., None] - s[:, 0]) ** 2 + (y[..., None] - s[:, 1]) ** 2
        return np.argmin(d2, axis=-1)

    for _ in range(int(relax)):
        lab = _labels(seeds)
        for k in range(n_grains):
            msk = (lab == k) & inside
            if msk.any():
                seeds[k] = [x[msk].mean(), y[msk].mean()]

    labels = np.where(inside, _labels(seeds), -1)
    angles = rng.uniform(0.0, np.pi, n_grains)
    theta_map = np.where(labels >= 0, angles[np.clip(labels, 0, None)], 0.0)
    return labels, angles, theta_map


def voronoi_polycrystal_3d(shape, n_grains, radius_m, h, rng=None, relax=1):
    """3D Voronoi polycrystal filling a cylindrical specimen.

    The full-height cylinder of the given radius is tessellated into
    ``n_grains`` 3D Voronoi cells; each grain gets a random 3D c-axis unit
    vector (uniform on the upper hemisphere). Same construction as the 2D
    version: nearest-seed assignment plus Lloyd relaxation.

    Returns
    -------
    labels : (nz, ny, nx) int ndarray
        Grain index per cell, -1 outside the specimen cylinder.
    axes : (n_grains, 3) ndarray
        Unit c-axis vector per grain.
    colat_map : (nz, ny, nx) ndarray
        Per-cell c-axis colatitude (angle to the cylinder axis; 0 outside).
    """
    if rng is None:
        rng = np.random.default_rng()
    nz, ny, nx = shape
    c = (nx - 1) * h / 2.0
    zmax = (nz - 1) * h
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float) * h
    inside = np.hypot(x - c, y - c) <= radius_m

    seeds = []
    while len(seeds) < n_grains:
        p = rng.uniform(-radius_m, radius_m, size=2)
        if np.hypot(*p) <= radius_m:
            seeds.append([p[0] + c, p[1] + c, rng.uniform(0.0, zmax)])
    seeds = np.asarray(seeds)

    def _labels(s):
        d2 = ((x[..., None] - s[:, 0]) ** 2 + (y[..., None] - s[:, 1]) ** 2
              + (z[..., None] - s[:, 2]) ** 2)
        return np.argmin(d2, axis=-1)

    for _ in range(int(relax)):
        lab = _labels(seeds)
        for k in range(n_grains):
            msk = (lab == k) & inside
            if msk.any():
                seeds[k] = [x[msk].mean(), y[msk].mean(), z[msk].mean()]

    labels = np.where(inside, _labels(seeds), -1)
    v = rng.standard_normal((n_grains, 3))
    v[:, 2] = np.abs(v[:, 2])                       # upper hemisphere
    axes = v / np.linalg.norm(v, axis=1, keepdims=True)
    colat = np.arccos(np.clip(axes[:, 2], -1.0, 1.0))
    colat_map = np.where(labels >= 0, colat[np.clip(labels, 0, None)], 0.0)
    return labels, axes, colat_map


# --- 3D phantoms ------------------------------------------------------------

def cylinder_background(shape, c_specimen, c_couplant, radius_m, h):
    """Cylindrical specimen along the z axis in a couplant bath (3D velocity).

    ``shape`` is (nz, ny, nx). The specimen is a cylinder of the given radius
    about the domain axis; the surrounding couplant fills the rest.
    """
    nz, ny, nx = shape
    cx = (nx - 1) * h / 2.0
    cy = (ny - 1) * h / 2.0
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float) * h
    r = np.hypot(x - cx, y - cy)
    return np.where(r <= radius_m, c_specimen, c_couplant)


def sphere_background(shape, c_specimen, c_couplant, radius_m, h):
    """Spherical specimen in a couplant bath (3D velocity), for shell arrays."""
    nz, ny, nx = shape
    cx = (nx - 1) * h / 2.0
    cy = (ny - 1) * h / 2.0
    cz = (nz - 1) * h / 2.0
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float) * h
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    return np.where(r <= radius_m, c_specimen, c_couplant)


def add_sphere(c, centre_frac, radius_m, c_inclusion, h):
    """Add a spherical velocity inclusion (a 3D flaw) to a velocity volume.

    ``centre_frac`` is (fx, fy, fz) in [0, 1] domain fractions.
    """
    nz, ny, nx = c.shape
    cx = centre_frac[0] * (nx - 1) * h
    cy = centre_frac[1] * (ny - 1) * h
    cz = centre_frac[2] * (nz - 1) * h
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float) * h
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    out = c.copy()
    out[r <= radius_m] = c_inclusion
    return out
