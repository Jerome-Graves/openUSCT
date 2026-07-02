"""Gradient check for the global-correlation-norm (GCN) misfit.

The adjoint-state gradient of the normalised-cross-correlation misfit must match
a central finite-difference estimate, exactly as for the L2 misfit. This
verifies the GCN adjoint source is correct.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def test_gcn_gradient_matches_fd():
    h = 1.0e-3
    ring = RingArray(n_elements=8, radius_m=0.024, domain_m=0.06, h=h)
    n = ring.n
    dt = 1.2e-7
    nt = 350
    wav = ricker(nt, dt, 0.4e6)
    src = [0, 4]

    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.5, 0.5), 0.006, 3400.0, h)
    dobs = fwi.forward_fmc(phantom.velocity_to_m(c_true), ring, wav, dt, h, nt, src_list=src)

    m0 = phantom.velocity_to_m(phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h))
    _, g = fwi.misfit_and_gradient(m0, ring, wav, dt, h, nt, dobs, src_list=src, misfit_type="gcn")

    c = n // 2
    probes = [(c, c), (c + 5, c), (c, c - 6)]
    rel = []
    for (iy, ix) in probes:
        eps = 1e-3 * m0[iy, ix]
        mp = m0.copy(); mp[iy, ix] += eps
        mm = m0.copy(); mm[iy, ix] -= eps
        Jp = fwi.misfit(mp, ring, wav, dt, h, nt, dobs, src_list=src, misfit_type="gcn")
        Jm = fwi.misfit(mm, ring, wav, dt, h, nt, dobs, src_list=src, misfit_type="gcn")
        fd = (Jp - Jm) / (2 * eps)
        ad = g[iy, ix]
        r = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
        rel.append(r)
        print(f"probe ({iy},{ix}): fd={fd:+.6e} adj={ad:+.6e} rel_err={r:.4f}")
    assert max(rel) < 0.05, f"GCN gradient mismatch: {rel}"


if __name__ == "__main__":
    test_gcn_gradient_matches_fd()
    print("GCN gradient check passed")
