"""3D adjoint gradient check.

The same discrete adjoint that was verified in 2D must also hold in 3D, since
the solver and gradient are dimension-general. This runs a small cubic problem
with a cylindrical array and confirms the adjoint-state gradient matches a
central finite-difference estimate at several interior voxels.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


def _setup():
    h = 1.5e-3
    domain = 0.03
    geom = CylinderArray(n_rings=2, per_ring=6, radius_m=0.010,
                         height_m=0.012, domain_m=domain, h=h)
    n = geom.n
    dt = 1.5e-7
    nt = 140
    wavelet = ricker(nt, dt, 0.35e6)

    c_true = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.008, h)
    c_true = phantom.add_sphere(c_true, (0.5, 0.5, 0.5), 0.004, 2700.0, h)
    m_true = phantom.velocity_to_m(c_true)

    c0 = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.008, h)
    m0 = phantom.velocity_to_m(c0)

    src_list = [0, 6]
    dobs = fwi.forward_fmc(m_true, geom, wavelet, dt, h, nt, src_list=src_list)
    return geom, wavelet, dt, h, nt, dobs, src_list, m0


def test_adjoint_matches_finite_difference_3d():
    geom, wavelet, dt, h, nt, dobs, src_list, m0 = _setup()

    _, g = fwi.misfit_and_gradient(m0, geom, wavelet, dt, h, nt, dobs, src_list=src_list)

    c = geom.n // 2
    probes = [(c, c, c), (c + 3, c, c), (c, c + 3, c), (c, c, c - 3)]

    rel_errors = []
    for p in probes:
        eps = 1.0e-3 * m0[p]
        mp = m0.copy(); mp[p] += eps
        mm = m0.copy(); mm[p] -= eps
        Jp = fwi.misfit(mp, geom, wavelet, dt, h, nt, dobs, src_list=src_list)
        Jm = fwi.misfit(mm, geom, wavelet, dt, h, nt, dobs, src_list=src_list)
        fd = (Jp - Jm) / (2.0 * eps)
        ad = g[p]
        denom = max(abs(fd), abs(ad), 1e-30)
        rel = abs(fd - ad) / denom
        rel_errors.append(rel)
        print(f"probe {p}: fd={fd:+.6e} adj={ad:+.6e} rel_err={rel:.4f}")

    assert max(rel_errors) < 0.05, f"3D gradient mismatch: {rel_errors}"


if __name__ == "__main__":
    test_adjoint_matches_finite_difference_3d()
    print("3D gradient check passed")
