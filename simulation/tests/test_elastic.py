"""Verify the elastic solver by measuring P and S wave speeds.

On the x-axis, an x-directed body force radiates a pure P wave (in vx) and a
y-directed force a pure S wave (in vy). Measuring the arrival times recovers
Vp = sqrt((lambda+2mu)/rho) and Vs = sqrt(mu/rho), which must match the values
the medium was built from.
"""

from __future__ import annotations

import numpy as np

from ringfwi import elastic
from ringfwi.sources import ricker


def _speed(source, record, vp0, vs0, rho0):
    """Differential travel time between two receivers (cancels source offset)."""
    n = 201
    h = 1.0e-3
    dt = 1.2e-7
    nt = 420
    f0 = 0.2e6
    wav = ricker(nt, dt, f0)

    vp = np.full((n, n), vp0)
    vs = np.full((n, n), vs0)
    rho = np.full((n, n), rho0)

    src = (100, 100)
    d1, d2 = 20, 50                      # receiver offsets in cells
    rec = [(100, 100 + d1), (100, 100 + d2)]

    trace, _ = elastic.forward(vp, vs, rho, h, dt, nt, src, wav, rec,
                               source=source, record=record)
    t1 = int(np.argmax(np.abs(trace[:, 0]))) * dt
    t2 = int(np.argmax(np.abs(trace[:, 1]))) * dt
    return (d2 - d1) * h / (t2 - t1)


def test_p_and_s_speeds():
    vp0, vs0, rho0 = 3000.0, 1500.0, 1000.0
    vp_meas = _speed("fx", "vx", vp0, vs0, rho0)   # x-force -> P on x-axis
    vs_meas = _speed("fy", "vy", vp0, vs0, rho0)   # y-force -> S on x-axis
    print(f"Vp: measured {vp_meas:.0f} m/s, expected {vp0:.0f} ({100*(vp_meas-vp0)/vp0:+.1f}%)")
    print(f"Vs: measured {vs_meas:.0f} m/s, expected {vs0:.0f} ({100*(vs_meas-vs0)/vs0:+.1f}%)")
    assert abs(vp_meas - vp0) / vp0 < 0.05
    assert abs(vs_meas - vs0) / vs0 < 0.05


if __name__ == "__main__":
    test_p_and_s_speeds()
    print("elastic P/S speed check passed")
