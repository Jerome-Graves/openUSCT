"""Gradient checks for multi-parameter (speed + attenuation) FWI.

Both the sound-speed gradient (g_m) and the attenuation gradient (g_a) from the
adjoint-state method must match central finite differences. This verifies the
exact discrete adjoint of the damped wave equation for both parameters.
"""

from __future__ import annotations

import numpy as np

from ringfwi import attenuation, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def _setup():
    h = 1.0e-3
    ring = RingArray(n_elements=8, radius_m=0.02, domain_m=0.05, h=h)
    n = ring.n
    dt = 1.5e-7
    nt = 300
    wav = ricker(nt, dt, 0.4e6)
    src = [0, 4]

    # True model: a velocity anomaly and an attenuation anomaly.
    c_true = np.full((n, n), 1500.0)
    c_true = phantom.add_inclusion(c_true, (0.55, 0.5), 0.006, 1800.0, h)
    m_true = phantom.velocity_to_m(c_true)
    a_true = np.full((n, n), 0.02)
    a_true = phantom.add_inclusion(a_true, (0.45, 0.55), 0.006, 0.08, h)

    dobs = attenuation.forward_fmc(m_true, a_true, ring, wav, dt, h, nt, src_list=src)

    m0 = phantom.velocity_to_m(np.full((n, n), 1500.0))
    a0 = np.full((n, n), 0.02)
    return ring, wav, dt, h, nt, dobs, src, m0, a0, n


def test_speed_and_attenuation_gradients():
    ring, wav, dt, h, nt, dobs, src, m0, a0, n = _setup()
    _, g_m, g_a = attenuation.misfit_and_gradients(m0, a0, ring, wav, dt, h, nt, dobs, src_list=src)

    c = n // 2
    probes = [(c, c), (c + 4, c), (c, c - 5)]

    # Sound-speed gradient.
    rel_m = []
    for (iy, ix) in probes:
        eps = 1e-3 * m0[iy, ix]
        mp = m0.copy(); mp[iy, ix] += eps
        mm = m0.copy(); mm[iy, ix] -= eps
        Jp = attenuation.misfit(mp, a0, ring, wav, dt, h, nt, dobs, src_list=src)
        Jm = attenuation.misfit(mm, a0, ring, wav, dt, h, nt, dobs, src_list=src)
        fd = (Jp - Jm) / (2 * eps)
        rel_m.append(abs(fd - g_m[iy, ix]) / max(abs(fd), abs(g_m[iy, ix]), 1e-30))
        print(f"g_m ({iy},{ix}): fd={fd:+.4e} adj={g_m[iy,ix]:+.4e} rel={rel_m[-1]:.4f}")

    # Attenuation gradient.
    rel_a = []
    for (iy, ix) in probes:
        eps = 1e-4
        ap = a0.copy(); ap[iy, ix] += eps
        am = a0.copy(); am[iy, ix] -= eps
        Jp = attenuation.misfit(m0, ap, ring, wav, dt, h, nt, dobs, src_list=src)
        Jm = attenuation.misfit(m0, am, ring, wav, dt, h, nt, dobs, src_list=src)
        fd = (Jp - Jm) / (2 * eps)
        rel_a.append(abs(fd - g_a[iy, ix]) / max(abs(fd), abs(g_a[iy, ix]), 1e-30))
        print(f"g_a ({iy},{ix}): fd={fd:+.4e} adj={g_a[iy,ix]:+.4e} rel={rel_a[-1]:.4f}")

    assert max(rel_m) < 0.05, f"speed gradient mismatch: {rel_m}"
    assert max(rel_a) < 0.05, f"attenuation gradient mismatch: {rel_a}"


if __name__ == "__main__":
    test_speed_and_attenuation_gradients()
    print("multi-parameter gradient checks passed")
