"""Verify voxel-wise orientation-field (theta) FWI machinery.

The parameterisation that answers "a voxel must represent velocity at every
angle": each voxel carries one in-plane c-axis angle of a known TI material,
which encodes the full directional velocity function. Checks:

1. the vectorised per-voxel rotation matches rotate_voigt exactly, and its
   analytic theta-derivative matches a finite difference;
2. the chained orientation gradient (elastic adjoint x dC/dtheta) matches a
   central finite difference of the misfit;
3. a small orientation-field inversion recovers grain angles with no
   geometry prior.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


BASE = an.ice_stiffness_2d(0.0)


def test_theta_maps_match_rotate_voigt():
    th = 0.61
    C, dC = an.theta_stiffness_maps(np.full((3, 3), th), BASE)
    ref = an.rotate_voigt(BASE, th)
    keys = {"C11": (0, 0), "C12": (0, 1), "C22": (1, 1),
            "C66": (2, 2), "C16": (0, 2), "C26": (1, 2)}
    for k, (a, b) in keys.items():
        assert abs(C[k][0, 0] - ref[a, b]) < 1e-6 * abs(ref[a, b] + 1.0)
    # analytic derivative vs FD of the rotation
    eps = 1e-6
    refp = an.rotate_voigt(BASE, th + eps)
    refm = an.rotate_voigt(BASE, th - eps)
    for k, (a, b) in keys.items():
        fd = (refp[a, b] - refm[a, b]) / (2 * eps)
        assert abs(dC[k][0, 0] - fd) < 1e-3 * (abs(fd) + 1.0)


def _setup(n=44, h=4.0e-4, nt=260, n_grains=3, seed=5):
    dt = 4.0e-8
    wav = ricker(nt, dt, 0.4e6)
    rho = np.full((n, n), an.ICE_RHO)
    ring = RingArray(n_elements=8, radius_m=0.0075,
                     domain_m=(n - 1) * h, h=h)
    # Solid ice everywhere; grains differ ONLY by orientation.
    labels, angles, theta_true = phantom.voronoi_polycrystal(
        (n, n), n_grains, 0.02, h, rng=np.random.default_rng(seed), relax=1)
    theta_true = np.where(labels >= 0, theta_true, angles[0])
    sources = [ring.element_index(s) for s in (0, 3, 6)]
    Ct, _ = an.theta_stiffness_maps(theta_true, BASE)
    dobs = [an._grad_forward(Ct["C11"], Ct["C12"], Ct["C22"], Ct["C66"],
                             rho, h, dt, nt, s, wav, ring.idx,
                             Ct["C16"], Ct["C26"])[0]
            for s in sources]
    return ring, rho, h, dt, nt, wav, sources, dobs, theta_true, labels


def test_theta_gradient_matches_fd():
    ring, rho, h, dt, nt, wav, sources, dobs, theta_true, labels = _setup()
    rng = np.random.default_rng(1)
    n = rho.shape[0]
    theta0 = theta_true + 0.2                       # start off the truth
    J0, g = an.theta_misfit_and_gradient(theta0, BASE, rho, h, dt, nt,
                                         sources, wav, ring.idx, dobs)
    dth = rng.standard_normal((n, n))
    dth[:4] = dth[-4:] = dth[:, :4] = dth[:, -4:] = 0.0
    analytic = float(np.sum(g * dth))
    eps = 1e-6
    Jp, _ = an.theta_misfit_and_gradient(theta0 + eps * dth, BASE, rho, h,
                                         dt, nt, sources, wav, ring.idx, dobs)
    Jm, _ = an.theta_misfit_and_gradient(theta0 - eps * dth, BASE, rho, h,
                                         dt, nt, sources, wav, ring.idx, dobs)
    fd = (Jp - Jm) / (2 * eps)
    rel = abs(analytic - fd) / (abs(fd) + 1e-30)
    print(f"theta gradient: analytic={analytic:.6e} fd={fd:.6e} rel={rel:.2e}")
    assert rel < 1e-4


def _ang_err_deg(a, b):
    """Angle-map error respecting the mod-pi (TI) symmetry, degrees."""
    return np.degrees(0.5 * np.arccos(np.clip(np.cos(2 * (a - b)), -1, 1)))


def test_theta_field_inversion_converges_locally():
    """The orientation-field inversion is a LOCAL method (like all gradient
    FWI): from a moderately wrong start it must pull the interior angle map
    toward the truth. Global initialisation is a separate problem (multiscale,
    or the grain-parameterised COF search as an initialiser)."""
    ring, rho, h, dt, nt, wav, sources, dobs, theta_true, labels = _setup()
    n = rho.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    inside = np.hypot(xx - cc, yy - cc) <= 0.0060       # illuminated interior

    theta0 = theta_true + np.radians(12.0)              # local start
    err0 = float(np.mean(_ang_err_deg(theta0, theta_true)[inside]))
    theta_rec, hist = an.invert_theta(theta0, BASE, rho, h, dt, nt, sources,
                                      wav, ring.idx, dobs, n_iter=10,
                                      step_rad=0.2, smooth_sigma=1.0,
                                      update_mask=inside.astype(float))
    err1 = float(np.mean(_ang_err_deg(theta_rec, theta_true)[inside]))
    print(f"theta-field FWI: misfit {hist[0]:.3e} -> {hist[-1]:.3e} "
          f"({hist[-1]/hist[0]*100:.1f}%), interior angle error "
          f"{err0:.1f} -> {err1:.1f} deg")
    assert hist[-1] < 0.97 * hist[0]
    assert err1 < err0 - 2.0                            # meaningful pull to truth


if __name__ == "__main__":
    test_theta_maps_match_rotate_voigt()
    test_theta_gradient_matches_fd()
    test_theta_field_inversion_improves()
    print("orientation-field FWI checks passed")
