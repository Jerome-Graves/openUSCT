"""Adjoint gradient check for finite-aperture (sized) elements.

Repeats the FWI gradient check with elements modelled as finite apertures
(``geometry.build_footprints``) rather than single grid points, confirming the
adjoint stays exact when the source and receiver are spread over a footprint.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray, build_footprints
from ringfwi.sources import ricker


def test_aperture_adjoint_matches_finite_difference():
    h = 1.0e-3
    domain = 0.06
    geom = RingArray(n_elements=8, radius_m=0.024, domain_m=domain, h=h)
    n = geom.n
    dt = 1.2e-7
    nt = 350
    wavelet = ricker(nt, dt, 0.4e6)
    footprints = build_footprints(geom, width_m=0.003, shape="flat")

    # Confirm the apertures really are multi-point.
    assert max(len(idxs) for idxs, _ in footprints) > 1

    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.5, 0.5), 0.006, 3400.0, h)
    m_true = phantom.velocity_to_m(c_true)
    m0 = phantom.velocity_to_m(phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h))

    src_list = [0, 4]
    dobs = fwi.forward_fmc(m_true, geom, wavelet, dt, h, nt, src_list=src_list,
                           footprints=footprints)

    def mg(m):
        return fwi.misfit_and_gradient(m, geom, wavelet, dt, h, nt, dobs,
                                       src_list=src_list, footprints=footprints)

    _, g = mg(m0)
    c = n // 2
    probes = [(c, c), (c + 5, c), (c, c - 6), (c - 4, c + 4)]
    rel_errors = []
    for (iy, ix) in probes:
        eps = 1.0e-3 * m0[iy, ix]
        mp = m0.copy(); mp[iy, ix] += eps
        mm = m0.copy(); mm[iy, ix] -= eps
        fd = (mg(mp)[0] - mg(mm)[0]) / (2.0 * eps)
        ad = g[iy, ix]
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
        rel_errors.append(rel)
        print(f"probe ({iy},{ix}): fd={fd:+.6e} adj={ad:+.6e} rel_err={rel:.4f}")
    assert max(rel_errors) < 0.05, f"gradient mismatch: {rel_errors}"


if __name__ == "__main__":
    test_aperture_adjoint_matches_finite_difference()
    print("aperture gradient check passed")
