"""End-to-end benchmark: 3D FWI with the Python vs the C++ (uap) backend.

Runs the same inversion both ways and confirms they produce the same result,
then reports the wall-clock speed-up from injecting the C++ backend into the
optimiser.

Run:  python bench_backend.py     (after building _uap)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))
sys.path.insert(0, os.path.dirname(__file__))

import uap
from ringfwi import fwi, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


def main():
    h = 1.25e-3
    cyl = CylinderArray(n_rings=3, per_ring=12, radius_m=0.015, height_m=0.014,
                        domain_m=0.036, h=h)
    n = cyl.n
    dt = 1.2e-7
    nt = 300
    wavelet = ricker(nt, dt, 0.3e6)

    c_true = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.012, h)
    c_true = phantom.add_sphere(c_true, (0.56, 0.5, 0.5), 0.005, 2650.0, h)
    c_bg = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.012, h)

    src_list = list(range(0, cyl.n_elements, 3))
    dobs = uap.forward_fmc(phantom.velocity_to_m(c_true), cyl, wavelet, dt, h, nt, src_list=src_list)

    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    r = np.hypot(xx - cc, yy - cc)
    mask = (r <= 0.012 * 0.95).astype(float)
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2500.0))
    m0 = phantom.velocity_to_m(c_bg)

    common = dict(src_list=src_list, n_iter=6, step_frac=0.04,
                  update_mask=mask, m_bounds=m_bounds)

    t = time.time()
    m_py, h_py = fwi.invert(m0, cyl, wavelet, dt, h, nt, dobs, **common)
    t_py = time.time() - t

    t = time.time()
    m_cpp, h_cpp = fwi.invert(m0, cyl, wavelet, dt, h, nt, dobs, backend=uap, **common)
    t_cpp = time.time() - t

    err = np.max(np.abs(m_py - m_cpp)) / (np.max(np.abs(m_py)) + 1e-30)
    print(f"grid {n}^3, {len(src_list)} transmitters, {len(h_py)-1} iterations")
    print(f"final misfit  python {h_py[-1]:.4e}  C++ {h_cpp[-1]:.4e}")
    print(f"model agreement (max rel diff): {err:.2e}")
    print(f"wall clock: python {t_py:.1f} s, C++ backend {t_cpp:.1f} s, "
          f"speed-up {t_py/max(t_cpp,1e-6):.1f}x")
    assert err < 1e-9


if __name__ == "__main__":
    main()
