"""Verify Jacobian (Gauss-Newton / Levenberg-Marquardt) COF iterations.

Starting from grain orientations perturbed ~15-20 degrees from the truth, a few
LM iterations on the finite-difference Jacobian must drive the misfit down by
orders of magnitude and every c-axis to within ~1 degree — the quadratic local
convergence that grid-search refinement cannot reach.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import cof, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import gabor


def test_jacobian_iterations_converge():
    mat = an.ICE_MATERIAL
    radius = 0.010
    dom = 2 * radius + 0.008
    n_grid = 22
    h = dom / (n_grid - 1)
    ring = CylinderArray(n_rings=2, per_ring=6, radius_m=radius, height_m=0.010,
                         domain_m=dom, h=h)
    n = ring.n
    dt = 0.4 * h / (an.ti_max_speed(mat) * 1.02 * np.sqrt(3))
    nt = 300
    wav = gabor(nt, dt, 0.3e6, 0.6)
    src_list = [0, 5]

    labels, axes_true, _ = phantom.voronoi_polycrystal_3d(
        (n, n, n), 3, 0.008, h, rng=np.random.default_rng(11), relax=1)
    dobs = cof.fmc(labels, axes_true, ring, h, dt, nt, wav, src_list,
                   material=mat)
    residual = cof.make_residual(labels, ring, h, dt, nt, wav, src_list, dobs,
                                 material=mat)

    # Perturb every grain's axis by 15-20 degrees.
    rng = np.random.default_rng(4)
    p_true = cof.params_from_axes(axes_true)
    p0 = p_true + np.radians(rng.uniform(12.0, 18.0, p_true.shape)) \
        * rng.choice([-1.0, 1.0], p_true.shape)
    errs0 = [cof.axis_error_deg(a, b)
             for a, b in zip(cof.axes_from_params(p0), axes_true)]

    p_rec, hist = cof.jacobian_iterations(residual, p0, n_iter=5, verbose=True)
    axes_rec = cof.axes_from_params(p_rec)
    errs = [cof.axis_error_deg(a, b) for a, b in zip(axes_rec, axes_true)]
    print(f"start errors {np.round(errs0, 1)} deg -> recovered "
          f"{np.round(errs, 2)} deg | J {hist[0]:.3e} -> {hist[-1]:.3e}")
    assert hist[-1] < 1e-3 * hist[0]                 # orders-of-magnitude drop
    assert max(errs) < 1.0                            # sub-degree recovery


if __name__ == "__main__":
    test_jacobian_iterations_converge()
    print("Jacobian-iteration COF checks passed")
