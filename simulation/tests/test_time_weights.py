"""Verify the windowed-misfit (time_weights) machinery.

Unit weights must reproduce the unweighted misfit and gradient exactly, and a
real P-window must produce a finite, different gradient.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def test_unit_weights_are_identity():
    h = 1.0e-3
    geom = RingArray(n_elements=8, radius_m=0.024, domain_m=0.06, h=h)
    n = geom.n
    dt = 1.2e-7
    nt = 300
    wav = ricker(nt, dt, 0.4e6)
    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.5, 0.5), 0.006, 3400.0, h)
    m_true = phantom.velocity_to_m(c_true)
    m0 = phantom.velocity_to_m(
        phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h))
    src_list = [0, 4]
    dobs = fwi.forward_fmc(m_true, geom, wav, dt, h, nt, src_list=src_list)

    J0, g0 = fwi.misfit_and_gradient(m0, geom, wav, dt, h, nt, dobs,
                                     src_list=src_list)
    ones = [np.ones((nt, geom.n_elements)) for _ in src_list]
    J1, g1 = fwi.misfit_and_gradient(m0, geom, wav, dt, h, nt, dobs,
                                     src_list=src_list, time_weights=ones)
    assert abs(J0 - J1) < 1e-12 * max(J0, 1e-30)
    assert np.allclose(g0, g1)

    # A genuine P-window changes the misfit but stays finite and non-trivial.
    tw = fwi.p_window_weights(dobs, dt, 0.4e6)
    assert all(w.shape == (nt, geom.n_elements) for w in tw)
    assert all((w >= 0).all() and (w <= 1).all() for w in tw)
    J2, g2 = fwi.misfit_and_gradient(m0, geom, wav, dt, h, nt, dobs,
                                     src_list=src_list, time_weights=tw)
    assert np.isfinite(J2) and np.isfinite(g2).all()
    assert J2 < J0                      # the mute removes misfit energy
    assert not np.allclose(g0, g2)


if __name__ == "__main__":
    test_unit_weights_are_identity()
    print("time-weights checks passed")
