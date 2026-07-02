"""Verify the Voronoi polycrystal phantom and its anisotropic forward model.

Checks that the tessellation fills the specimen disc, that the per-grain
stiffness maps are physically valid (positive definite in the grains, fluid in
the couplant), and that a wave propagated through the polycrystal arrives at a
far receiver within the physical speed bounds of ice and water.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import gabor


def _setup(n=121, h=5.0e-4, n_grains=12, seed=3):
    rng = np.random.default_rng(seed)
    radius = 0.021
    labels, angles, theta = phantom.voronoi_polycrystal((n, n), n_grains, radius,
                                                        h, rng=rng, relax=1)
    return labels, angles, theta, radius, h


def test_tessellation_fills_disc():
    labels, angles, theta, radius, h = _setup()
    n = labels.shape[0]
    c = (n - 1) * h / 2.0
    y, x = np.mgrid[0:n, 0:n].astype(float) * h
    inside = np.hypot(x - c, y - c) <= radius
    assert (labels[inside] >= 0).all()            # every specimen cell has a grain
    assert (labels[~inside] == -1).all()          # couplant is unlabelled
    present = len(np.unique(labels[labels >= 0]))
    assert present >= 10                          # nearly all 12 grains survive
    assert (theta[inside] >= 0).all() and (theta[inside] < np.pi).all()


def test_stiffness_maps_valid():
    labels, angles, theta, radius, h = _setup()
    (C11, C12, C22, C16, C26, C66), rho = an.polycrystal_stiffness(labels, angles)

    ice = labels >= 0
    # In-plane normal stiffness block positive definite in every grain cell.
    det = C11[ice] * C22[ice] - C12[ice] ** 2
    assert (C11[ice] > 0).all() and (det > 0).all() and (C66[ice] > 0).all()
    assert np.isclose(rho[ice].mean(), an.ICE_RHO)
    # Rotated grains activate the off-diagonal coupling somewhere.
    assert np.max(np.abs(C16[ice])) > 1e8
    # Couplant is fluid: zero shear, water bulk stiffness.
    coup = labels < 0
    assert np.allclose(C66[coup], 0.0)
    assert np.allclose(C11[coup], 1000.0 * 1480.0 ** 2)
    # Apparent-speed map spans the ice qP range (the surface dips to ~3776 m/s
    # at intermediate c-axis angles) and water outside.
    capp = an.polycrystal_apparent_speed(labels, angles)
    assert 3700.0 < capp[ice].min() and capp[ice].max() < 4100.0
    assert capp[ice].max() - capp[ice].min() > 50.0    # real orientation contrast
    assert np.allclose(capp[coup], 1480.0)


def test_wave_through_polycrystal():
    labels, angles, theta, radius, h = _setup()
    n = labels.shape[0]
    Cmaps, rho = an.polycrystal_stiffness(labels, angles)

    ring = RingArray(n_elements=8, radius_m=0.024, domain_m=(n - 1) * h, h=h)
    dt = 0.4 * h / (4100.0 * np.sqrt(2.0))
    nt = 700
    f0 = 0.4e6
    wav = gabor(nt, dt, f0, 0.6)
    sigma_t = 1.0 / (2.0 * np.pi * (0.6 * f0 / 2.3548))
    t0 = 3.0 * sigma_t                                  # gabor peak time

    src = ring.element_index(0)
    far = ring.element_index(4)                         # diametrically opposite
    rec, _ = an.forward(Cmaps, rho, h, dt, nt, src, wav, [far],
                        source="explosive", record="pressure")
    assert np.isfinite(rec).all() and np.abs(rec).max() > 0

    d = 2.0 * ring.radius_m
    t_peak = int(np.argmax(np.abs(rec[:, 0]))) * dt
    v_app = d / (t_peak - t0)
    print(f"polycrystal apparent speed src->far: {v_app:.0f} m/s")
    # Path crosses couplant gaps + ice grains: between water and max ice qP.
    assert 1400.0 < v_app < 4300.0


if __name__ == "__main__":
    test_tessellation_fills_disc()
    test_stiffness_maps_valid()
    test_wave_through_polycrystal()
    print("polycrystal checks passed")
