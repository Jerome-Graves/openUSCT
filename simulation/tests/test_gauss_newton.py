"""Verify the Gauss-Newton ("Jacobian iteration") FWI machinery.

1. The matrix-free Hessian-vector product (J^T J) v — Born forward composed
   with the exact adjoint — must match a finite difference of the gradient at
   a zero-residual point, where FD-of-gradient equals the exact Gauss-Newton
   Hessian.
2. Gauss-Newton outer iterations must reduce the misfit far faster than the
   same number of gradient-descent iterations.
"""

from __future__ import annotations

import numpy as np

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def _setup():
    h = 1.0e-3
    geom = RingArray(n_elements=8, radius_m=0.024, domain_m=0.06, h=h)
    n = geom.n
    dt = 1.2e-7
    nt = 300
    wav = ricker(nt, dt, 0.4e6)
    return geom, n, h, dt, nt, wav


def test_hessian_vector_matches_fd():
    geom, n, h, dt, nt, wav = _setup()
    m0 = phantom.velocity_to_m(
        phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h))
    sl = [0, 4]
    dobs = fwi.forward_fmc(m0, geom, wav, dt, h, nt, src_list=sl)  # r = 0 at m0

    rng = np.random.default_rng(2)
    v = rng.standard_normal((n, n))
    v[:5] = v[-5:] = v[:, :5] = v[:, -5:] = 0.0
    v *= float(np.mean(m0))

    Hv = fwi.gn_hessian_vector(m0, geom, wav, dt, h, nt, v, src_list=sl)
    eps = 1e-4
    _, gp = fwi.misfit_and_gradient(m0 + eps * v, geom, wav, dt, h, nt, dobs,
                                    src_list=sl)
    _, gm = fwi.misfit_and_gradient(m0 - eps * v, geom, wav, dt, h, nt, dobs,
                                    src_list=sl)
    fd = (gp - gm) / (2.0 * eps)
    rel = np.max(np.abs(Hv - fd)) / (np.max(np.abs(fd)) + 1e-30)
    print(f"GN Hessian-vector vs FD-of-gradient: rel err {rel:.2e}")
    assert rel < 1e-5


def test_gauss_newton_beats_gradient_descent():
    geom, n, h, dt, nt, wav = _setup()
    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.55, 0.5), 0.006, 3350.0, h)
    m_true = phantom.velocity_to_m(c_true)
    c0 = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    m0 = phantom.velocity_to_m(c0)
    sl = [0, 2, 4, 6]
    dobs = fwi.forward_fmc(m_true, geom, wav, dt, h, nt, src_list=sl)

    yy, xx = np.mgrid[0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    mask = (np.hypot(xx - cc, yy - cc) <= 0.019).astype(float)

    m_gd, hist_gd = fwi.invert(m0, geom, wav, dt, h, nt, dobs, src_list=sl,
                               n_iter=4, update_mask=mask)
    m_gn, hist_gn = fwi.invert(m0, geom, wav, dt, h, nt, dobs, src_list=sl,
                               n_iter=4, update_mask=mask,
                               optimizer="gauss-newton", n_cg=8)
    red_gd = hist_gd[-1] / hist_gd[0]
    red_gn = hist_gn[-1] / hist_gn[0]
    print(f"4 iterations: GD misfit -> {red_gd*100:.2f}% | "
          f"GN misfit -> {red_gn*100:.3f}%")
    assert red_gn < red_gd            # curvature-aware step wins
    assert red_gn < 0.05              # and converges hard


if __name__ == "__main__":
    test_hessian_vector_matches_fd()
    test_gauss_newton_beats_gradient_descent()
    print("Gauss-Newton checks passed")
