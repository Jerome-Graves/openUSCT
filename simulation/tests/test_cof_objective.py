"""Concept check for grain-orientation (COF) inversion.

The full-waveform objective — elastic FMC data misfit as a function of the
per-grain c-axis angles with known geometry and material — must be exactly
zero at the true orientations and grow monotonically with orientation error.
That sensitivity is what makes the grain-parameterised inversion well posed.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import elastic3d, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import gabor


def _fmc(labels, axes, ring, h, dt, nt, wav, src_list, mat):
    Cm, rho = an.polycrystal_stiffness_3d(labels, axes, material=mat)
    data = np.zeros((len(src_list), nt, ring.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = elastic3d.forward(Cm, rho, h, dt, nt, ring.element_index(s),
                                   wav, ring.idx, source="explosive",
                                   record="pressure")
        data[i] = rec
    return data


def _rotate_axis(a, deg):
    """Tilt a unit axis by ``deg`` about a perpendicular direction."""
    a = np.asarray(a, float)
    t = np.cross(a, [1.0, 0.3, 0.2])
    t /= np.linalg.norm(t)
    r = np.radians(deg)
    out = np.cos(r) * a + np.sin(r) * t
    return out / np.linalg.norm(out)


def test_objective_zero_at_truth_and_monotone():
    mat = an.ICE_MATERIAL
    radius = 0.010
    dom = 2 * radius + 0.008
    n_grid = 22
    h = dom / (n_grid - 1)
    ring = CylinderArray(n_rings=2, per_ring=6, radius_m=radius, height_m=0.010,
                         domain_m=dom, h=h)
    n = ring.n
    dt = 0.4 * h / (an.ti_max_speed(mat) * 1.02 * np.sqrt(3))
    nt = 300
    wav = gabor(nt, dt, 0.3e6, 0.6)
    src_list = [0, 5]

    labels, axes_true, _ = phantom.voronoi_polycrystal_3d(
        (n, n, n), 3, 0.008, h, rng=np.random.default_rng(11), relax=1)
    dobs = _fmc(labels, axes_true, ring, h, dt, nt, wav, src_list, mat)
    # Per-trace normalised L2 with the transmitter's own trace excluded: the
    # near-source blast is orientation-insensitive and otherwise swamps J.
    tr_norm = np.sum(dobs ** 2, axis=1) + 1e-30       # (n_tx, n_rx)
    w = np.ones_like(tr_norm)
    for i, s in enumerate(src_list):
        w[i, s] = 0.0

    def J(axes):
        d = _fmc(labels, axes, ring, h, dt, nt, wav, src_list, mat)
        return float(np.sum(w * np.sum((d - dobs) ** 2, axis=1) / tr_norm))

    assert J(axes_true) < 1e-20                       # exact model -> zero misfit

    j_prev = 0.0
    for deg in (5.0, 10.0, 25.0, 45.0):
        ax = axes_true.copy()
        ax[0] = _rotate_axis(ax[0], deg)              # tilt one grain
        j = J(ax)
        print(f"grain-0 tilt {deg:.0f} deg -> J = {j:.3e}")
        assert j > j_prev                             # monotone sensitivity
        j_prev = j
    assert j_prev > 0.1                               # strongly sensitive


if __name__ == "__main__":
    test_objective_zero_at_truth_and_monotone()
    print("COF objective checks passed")
