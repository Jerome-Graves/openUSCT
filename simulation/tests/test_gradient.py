"""Adjoint gradient check.

Verifies that the adjoint-state gradient produced by
``fwi.misfit_and_gradient`` agrees with a central finite-difference estimate of
the misfit gradient at a handful of grid points. Agreement to a few percent
confirms the discrete adjoint is implemented correctly, which is the whole
credibility of an FWI code.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.solver import make_sponge
from ringfwi.sources import ricker


def _small_setup():
    h = 1.0e-3
    domain = 0.06
    geom = RingArray(n_elements=8, radius_m=0.024, domain_m=domain, h=h)
    n = geom.n
    dt = 1.2e-7
    nt = 350
    f0 = 0.4e6
    wavelet = ricker(nt, dt, f0)
    sponge = None  # exact discrete adjoint is validated without boundary damping

    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.5, 0.5), 0.006, 3400.0, h)
    m_true = phantom.velocity_to_m(c_true)

    c_start = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    m0 = phantom.velocity_to_m(c_start)

    src_list = [0, 4]
    dobs = fwi.forward_fmc(m_true, geom, wavelet, dt, h, nt, sponge, src_list)
    return geom, wavelet, dt, h, nt, dobs, sponge, src_list, m0


def test_adjoint_matches_finite_difference():
    geom, wavelet, dt, h, nt, dobs, sponge, src_list, m0 = _small_setup()

    J0, g = fwi.misfit_and_gradient(m0, geom, wavelet, dt, h, nt, dobs, sponge, src_list)

    # Probe a few interior points near the specimen centre.
    c = geom.n // 2
    probes = [(c, c), (c + 5, c), (c, c - 6), (c - 4, c + 4)]

    rel_errors = []
    for (iy, ix) in probes:
        eps = 1.0e-3 * m0[iy, ix]

        mp = m0.copy()
        mp[iy, ix] += eps
        Jp, _ = fwi.misfit_and_gradient(mp, geom, wavelet, dt, h, nt, dobs, sponge, src_list)

        mm = m0.copy()
        mm[iy, ix] -= eps
        Jm, _ = fwi.misfit_and_gradient(mm, geom, wavelet, dt, h, nt, dobs, sponge, src_list)

        fd = (Jp - Jm) / (2.0 * eps)
        ad = g[iy, ix]
        denom = max(abs(fd), abs(ad), 1e-30)
        rel = abs(fd - ad) / denom
        rel_errors.append(rel)
        print(f"probe ({iy},{ix}): fd={fd:+.6e} adj={ad:+.6e} rel_err={rel:.4f}")

    assert max(rel_errors) < 0.05, f"gradient mismatch: {rel_errors}"


if __name__ == "__main__":
    test_adjoint_matches_finite_difference()
    print("gradient check passed")
