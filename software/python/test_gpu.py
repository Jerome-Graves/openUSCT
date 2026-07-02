"""Verify and benchmark the CuPy GPU backend against the CPU cores.

The GPU runs in FP32, so it agrees with the FP64 CPU reference to about 1e-3
(the tolerance FWI needs), not to machine precision. This checks that, then
benchmarks the GPU against the OpenMP C++ CPU core on a large grid, and runs a
full inversion on the GPU backend end to end.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import pytest
    pytest.importorskip("cupy")
except ImportError:
    pass

import uap        # C++ CPU core
import uap_gpu    # CuPy GPU core
from ringfwi import fwi, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


def _problem(per_ring, nrings, radius, height, domain, h, nt, dt):
    cyl = CylinderArray(n_rings=nrings, per_ring=per_ring, radius_m=radius,
                        height_m=height, domain_m=domain, h=h)
    n = cyl.n
    wav = ricker(nt, dt, 0.3e6)
    c = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, radius * 0.8, h)
    c = phantom.add_sphere(c, (0.56, 0.5, 0.5), 0.005, 2650.0, h)
    return cyl, n, wav, phantom.velocity_to_m(c)


def test_gpu_parity_and_speed():
    # --- parity vs FP64 CPU reference (moderate grid) ----------------------
    cyl, n, wav, m = _problem(12, 3, 0.015, 0.014, 0.036, 1.25e-3, 250, 1.2e-7)
    src = list(range(0, cyl.n_elements, 3))
    dt, h = 1.2e-7, 1.25e-3
    dobs = uap.forward_fmc(m, cyl, wav, dt, h, 250, src_list=src)
    m0 = phantom.velocity_to_m(np.full((n, n, n), 3000.0))

    J_cpu, g_cpu = fwi.misfit_and_gradient(m0, cyl, wav, dt, h, 250, dobs, src_list=src)
    J_gpu, g_gpu = uap_gpu.misfit_and_gradient(m0, cyl, wav, dt, h, 250, dobs, src_list=src)
    J_err = abs(J_cpu - J_gpu) / abs(J_cpu)
    g_err = np.max(np.abs(g_cpu - g_gpu)) / np.max(np.abs(g_cpu))
    print(f"[parity] FP32 GPU vs FP64 CPU: misfit relerr {J_err:.2e}, gradient relerr {g_err:.2e}")
    assert J_err < 5e-3 and g_err < 5e-2

    # --- benchmark on a large grid: OpenMP C++ vs GPU ----------------------
    cyl, n, wav, m = _problem(16, 3, 0.020, 0.020, 0.048, 1.0e-3, 300, 1.0e-7)
    src = list(range(0, cyl.n_elements, 4))
    dt, h = 1.0e-7, 1.0e-3
    dobs = uap.forward_fmc(m, cyl, wav, dt, h, 300, src_list=src)

    uap_gpu.misfit_and_gradient(m, cyl, wav, dt, h, 300, dobs, src_list=src)  # warm up JIT
    import cupy as cp

    t = time.time(); uap.misfit_and_gradient(m, cyl, wav, dt, h, 300, dobs, src_list=src); t_cpu = time.time() - t
    t = time.time(); uap_gpu.misfit_and_gradient(m, cyl, wav, dt, h, 300, dobs, src_list=src); cp.cuda.Stream.null.synchronize(); t_gpu = time.time() - t
    print(f"[bench] grid {n}^3, misfit+gradient: OpenMP C++ {t_cpu*1e3:.0f} ms, "
          f"GPU {t_gpu*1e3:.0f} ms, speed-up {t_cpu/max(t_gpu,1e-6):.1f}x")


def test_gpu_end_to_end():
    cyl, n, wav, m = _problem(12, 3, 0.015, 0.014, 0.036, 1.25e-3, 300, 1.2e-7)
    src = list(range(0, cyl.n_elements, 3))
    dt, h = 1.2e-7, 1.25e-3
    dobs = uap.forward_fmc(m, cyl, wav, dt, h, 300, src_list=src)

    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    mask = (np.hypot(xx - cc, yy - cc) <= 0.012 * 0.95).astype(float)
    c_bg = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.015 * 0.8, h)
    m0 = phantom.velocity_to_m(c_bg)  # correct background, no flaw
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2500.0))

    m_rec, hist = fwi.invert(m0, cyl, wav, dt, h, 300, dobs, src_list=src,
                             n_iter=6, step_frac=0.04, update_mask=mask,
                             m_bounds=m_bounds, backend=uap_gpu)
    print(f"[e2e] GPU-backend FWI: misfit {hist[0]:.3e} -> {hist[-1]:.3e} "
          f"({hist[-1]/hist[0]*100:.1f}%)")
    assert hist[-1] < 0.5 * hist[0]


if __name__ == "__main__":
    test_gpu_parity_and_speed()
    test_gpu_end_to_end()
    print("GPU backend verified")
