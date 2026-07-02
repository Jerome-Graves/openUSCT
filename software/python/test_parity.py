"""Parity and benchmark: libuap C++ core vs the Python reference.

The C++ core in libuap.hpp mirrors ringfwi.fwi op for op, so on the same inputs
it must produce the same full-matrix-capture data, misfit, and gradient to
machine precision. This test proves that in 2D and 3D, and reports the speed-up.

Run:  python test_parity.py     (after: python setup.py build_ext --inplace)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))
sys.path.insert(0, os.path.dirname(__file__))

import _uap  # the compiled C++ core
from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray, CylinderArray
from ringfwi.sources import ricker


def linear_indices(idx, shape):
    """Flatten grid index tuples to row-major linear indices."""
    strides = np.cumprod((1,) + shape[::-1][:-1])[::-1]
    return np.array([int(np.dot(np.array(t), strides)) for t in idx], dtype=np.int32)


def _check(geom, c_field, dt, nt, f0, tag):
    shape = c_field.shape
    wavelet = ricker(nt, dt, f0)
    m = phantom.velocity_to_m(c_field)
    h = geom.h

    rec_lin = linear_indices(geom.idx, shape)
    tx_lin = rec_lin.copy()

    # Forward parity.
    d_py = fwi.forward_fmc(m, geom, wavelet, dt, h, nt)
    d_cpp = _uap.forward_fmc(m, h, dt, nt, tx_lin, rec_lin, wavelet)
    fwd_err = np.max(np.abs(d_py - d_cpp)) / (np.max(np.abs(d_py)) + 1e-30)

    # Gradient parity (use the forward data as the target so residuals are nonzero
    # via a perturbed starting model).
    m0 = phantom.velocity_to_m(np.full(shape, np.mean(c_field)))
    J_py, g_py = fwi.misfit_and_gradient(m0, geom, wavelet, dt, h, nt, d_py)
    J_cpp, g_cpp = _uap.misfit_and_gradient(m0, h, dt, nt, tx_lin, rec_lin, wavelet, d_py)
    J_err = abs(J_py - J_cpp) / (abs(J_py) + 1e-30)
    g_err = np.max(np.abs(g_py - g_cpp)) / (np.max(np.abs(g_py)) + 1e-30)

    print(f"[{tag}] forward relerr {fwd_err:.2e}, misfit relerr {J_err:.2e}, "
          f"gradient relerr {g_err:.2e}")
    assert fwd_err < 1e-9 and J_err < 1e-9 and g_err < 1e-9

    # Benchmark one misfit+gradient evaluation.
    t = time.time(); fwi.misfit_and_gradient(m0, geom, wavelet, dt, h, nt, d_py); t_py = time.time() - t
    t = time.time(); _uap.misfit_and_gradient(m0, h, dt, nt, tx_lin, rec_lin, wavelet, d_py); t_cpp = time.time() - t
    print(f"[{tag}] misfit+gradient: python {t_py*1e3:.0f} ms, C++ {t_cpp*1e3:.0f} ms, "
          f"speed-up {t_py/max(t_cpp,1e-6):.1f}x")


def test_parity_2d():
    ring = RingArray(n_elements=16, radius_m=0.024, domain_m=0.060, h=6e-4)
    n = ring.n
    c = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.020, 6e-4)
    c = phantom.add_inclusion(c, (0.58, 0.46), 0.006, 2650.0, 6e-4)
    _check(ring, c, dt=7e-8, nt=500, f0=0.3e6, tag="2D")


def test_parity_3d():
    cyl = CylinderArray(n_rings=3, per_ring=12, radius_m=0.015, height_m=0.014,
                        domain_m=0.036, h=1.25e-3)
    n = cyl.n
    c = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.012, 1.25e-3)
    c = phantom.add_sphere(c, (0.56, 0.5, 0.5), 0.005, 2650.0, 1.25e-3)
    _check(cyl, c, dt=1.2e-7, nt=250, f0=0.3e6, tag="3D")


if __name__ == "__main__":
    test_parity_2d()
    test_parity_3d()
    print("C++/Python parity verified")
