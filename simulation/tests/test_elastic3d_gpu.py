"""GPU (CuPy, float32) parity for the 3D anisotropic elastic solver.

Skipped when no CUDA device is available. The GPU path must reproduce the
float64 CPU reference to float32 precision on a polycrystal medium.
"""

from __future__ import annotations

import numpy as np
import pytest

from ringfwi import anisotropy as an
from ringfwi import elastic3d, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


@pytest.mark.skipif(not elastic3d.gpu_available(), reason="no CUDA GPU")
def test_gpu_matches_cpu():
    n = 36                                   # above the auto-GPU threshold
    h = 0.028 / (n - 1)
    labels, axes, _ = phantom.voronoi_polycrystal_3d(
        (n, n, n), 8, 0.008, h, rng=np.random.default_rng(3))
    Cm, rho = an.polycrystal_stiffness_3d(labels, axes)
    dt = 0.4 * h / (4100.0 * np.sqrt(3))
    nt = 200
    wav = ricker(nt, dt, 0.3e6)
    ring = CylinderArray(n_rings=2, per_ring=8, radius_m=0.010, height_m=0.012,
                         domain_m=0.028, h=h)
    src = ring.element_index(0)

    rc, _ = elastic3d.forward(Cm, rho, h, dt, nt, src, wav, ring.idx,
                              device="cpu")
    rg, _ = elastic3d.forward(Cm, rho, h, dt, nt, src, wav, ring.idx,
                              device="gpu")
    rel = np.max(np.abs(rc - rg)) / (np.max(np.abs(rc)) + 1e-30)
    print(f"GPU vs CPU parity: rel {rel:.2e}")
    assert rel < 1e-4

    # Footprint (finite-aperture) path on the GPU too.
    from ringfwi.geometry import build_footprints
    fp = build_footprints(ring, 0.002, "rect", height_m=0.003)
    src_pts = list(zip(fp[0][0], fp[0][1]))
    rc2, _ = elastic3d.forward(Cm, rho, h, dt, nt, None, wav, ring.idx,
                               src_pts=src_pts, rec_groups=fp, device="cpu")
    rg2, _ = elastic3d.forward(Cm, rho, h, dt, nt, None, wav, ring.idx,
                               src_pts=src_pts, rec_groups=fp, device="gpu")
    rel2 = np.max(np.abs(rc2 - rg2)) / (np.max(np.abs(rc2)) + 1e-30)
    print(f"GPU vs CPU parity (footprints): rel {rel2:.2e}")
    assert rel2 < 1e-4


if __name__ == "__main__":
    test_gpu_matches_cpu()
    print("elastic3d GPU parity passed")
