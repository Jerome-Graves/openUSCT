"""Verify the anisotropic (rotated-staggered-grid) elastic solver.

Checks, in order of increasing stringency:
  1. tensor-rotation machinery: isotropy is rotation-invariant, and the
     Christoffel equation reproduces the isotropic Vp/Vs;
  2. isotropic reduction of the RSG solver gives the correct Vp and Vs;
  3. ice Ih (orthotropic, c-axis on-axis): measured qP along x and y match the
     Christoffel phase velocities sqrt(C11/rho), sqrt(C22/rho);
  4. ice Ih with the c-axis rotated 30 deg (non-zero C16/C26): measured qP along
     x matches the Christoffel prediction, verifying the off-diagonal coupling.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi.sources import ricker


def _iso_voigt(vp, vs, rho):
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])


def _measure_qp(C, rho0, axis):
    """qP phase speed along a grid axis via differential travel time."""
    n = 181
    h = 3.0e-4
    dt = 3.0e-8
    nt = 360
    wav = ricker(nt, dt, 0.4e6)
    rho = np.full((n, n), rho0)
    Cmaps = (C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2])

    d1, d2 = 40, 80
    if axis == "x":
        src = (90, 45); rec = [(90, 45 + d1), (90, 45 + d2)]
    else:
        src = (45, 90); rec = [(45 + d1, 90), (45 + d2, 90)]

    trace, _ = an.forward(Cmaps, rho, h, dt, nt, src, wav, rec,
                          source="explosive", record="pressure")
    t1 = int(np.argmax(np.abs(trace[:, 0]))) * dt
    t2 = int(np.argmax(np.abs(trace[:, 1]))) * dt
    return (d2 - d1) * h / (t2 - t1)


def test_rotation_and_christoffel():
    C = _iso_voigt(3000.0, 1500.0, 1000.0)
    # isotropy invariant under rotation
    assert np.allclose(an.rotate_voigt(C, 0.7), C, atol=1.0)
    # Christoffel reproduces the isotropic speeds in every direction
    for phi in (0.0, 0.5, 1.1, 2.3):
        vqp, vqs = an.christoffel_velocities(C, 1000.0, phi)
        assert abs(vqp - 3000.0) < 3.0 and abs(vqs - 1500.0) < 3.0


def test_isotropic_reduction():
    C = _iso_voigt(3000.0, 1500.0, 1000.0)
    vp = _measure_qp(C, 1000.0, "x")
    print(f"iso staggered Vp: {vp:.0f} (expect 3000, {100*(vp-3000)/3000:+.1f}%)")
    assert abs(vp - 3000.0) / 3000.0 < 0.05


def test_ice_anisotropy_onaxis():
    C = an.ice_stiffness_2d(0.0)
    qp_x = _measure_qp(C, an.ICE_RHO, "x")
    qp_y = _measure_qp(C, an.ICE_RHO, "y")
    cx, _ = an.christoffel_velocities(C, an.ICE_RHO, 0.0)       # along x
    cy, _ = an.christoffel_velocities(C, an.ICE_RHO, np.pi / 2)  # along y (c-axis)
    print(f"ice qP along x: {qp_x:.0f} vs Christoffel {cx:.0f} "
          f"({100*(qp_x-cx)/cx:+.1f}%)")
    print(f"ice qP along y: {qp_y:.0f} vs Christoffel {cy:.0f} "
          f"({100*(qp_y-cy)/cy:+.1f}%)")
    print(f"P-anisotropy measured {100*(qp_y-qp_x)/qp_x:+.1f}%, "
          f"true {100*(cy-cx)/cx:+.1f}%")
    assert abs(qp_x - cx) / cx < 0.05
    assert abs(qp_y - cy) / cy < 0.05
    assert qp_y > qp_x   # stiffer along the c-axis


def test_ice_anisotropy_rotated():
    C = an.ice_stiffness_2d(np.radians(30.0))
    assert abs(C[0, 2]) > 1e8 and abs(C[1, 2]) > 1e8   # C16/C26 now active
    qp_x = _measure_qp(C, an.ICE_RHO, "x")
    cx, _ = an.christoffel_velocities(C, an.ICE_RHO, 0.0)
    print(f"ice(30deg) qP along x: {qp_x:.0f} vs Christoffel {cx:.0f} "
          f"({100*(qp_x-cx)/cx:+.1f}%)")
    assert abs(qp_x - cx) / cx < 0.05


if __name__ == "__main__":
    test_rotation_and_christoffel()
    test_isotropic_reduction()
    test_ice_anisotropy_onaxis()
    test_ice_anisotropy_rotated()
    print("anisotropy checks passed")
