"""Verify the full 3D anisotropic elastic solver against the 3D Christoffel
equation (and the isotropic limit).

Speeds are measured by differential travel time between two receivers on a
grid axis, as in the verified 2D tests.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import elastic3d
from ringfwi.sources import ricker


def _iso_voigt6(vp, vs, rho):
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    for i in range(3):
        C[i, i] = lam + 2 * mu
        C[3 + i, 3 + i] = mu
    return C


def _measure_qp(C6, rho0, n=56, h=6.0e-4, f0=0.5e6, axis="x"):
    """qP speed along a grid axis via two-receiver differential travel time."""
    Cmaps = {f"C{i}{j}": C6[i - 1, j - 1] for i in range(1, 7) for j in range(i, 7)}
    rho = np.full((n, n, n), rho0)
    vmax = an.christoffel_3d(C6, rho0, (1, 0, 0))[0] * 1.1
    dt = 0.4 * h / (vmax * np.sqrt(3))
    nt = 300
    wav = ricker(nt, dt, f0)

    c = n // 2
    d1, d2 = 10, 24
    if axis == "x":
        src = (c, c, 8); recs = [(c, c, 8 + d1), (c, c, 8 + d2)]
    else:                                   # z
        src = (8, c, c); recs = [(8 + d1, c, c), (8 + d2, c, c)]

    rec, _ = elastic3d.forward(Cmaps, rho, h, dt, nt, src, wav, recs,
                               source="explosive", record="pressure")
    t1 = int(np.argmax(np.abs(rec[:, 0]))) * dt
    t2 = int(np.argmax(np.abs(rec[:, 1]))) * dt
    return (d2 - d1) * h / (t2 - t1)


def test_isotropic_reduction():
    vp0, vs0, rho0 = 3000.0, 1500.0, 1000.0
    C6 = _iso_voigt6(vp0, vs0, rho0)
    vqp = an.christoffel_3d(C6, rho0, (1, 0, 0))
    assert abs(vqp[0] - vp0) < 2.0 and abs(vqp[1] - vs0) < 2.0
    vp = _measure_qp(C6, rho0)
    print(f"3D iso Vp: {vp:.0f} (expect {vp0:.0f}, {100*(vp-vp0)/vp0:+.1f}%)")
    assert abs(vp - vp0) / vp0 < 0.05


def test_ice_on_axis():
    C6 = an.ice_stiffness_3d((0, 0, 1))
    qp_x = _measure_qp(C6, an.ICE_RHO, axis="x")
    qp_z = _measure_qp(C6, an.ICE_RHO, axis="z")
    cx = an.christoffel_3d(C6, an.ICE_RHO, (1, 0, 0))[0]     # sqrt(C11/rho)
    cz = an.christoffel_3d(C6, an.ICE_RHO, (0, 0, 1))[0]     # sqrt(C33/rho)
    print(f"3D ice qP x: {qp_x:.0f} vs Christoffel {cx:.0f} ({100*(qp_x-cx)/cx:+.1f}%)")
    print(f"3D ice qP z: {qp_z:.0f} vs Christoffel {cz:.0f} ({100*(qp_z-cz)/cz:+.1f}%)")
    assert abs(cx - np.sqrt(an.ICE_C11 / an.ICE_RHO)) < 1.0
    assert abs(cz - np.sqrt(an.ICE_C33 / an.ICE_RHO)) < 1.0
    assert abs(qp_x - cx) / cx < 0.05
    assert abs(qp_z - cz) / cz < 0.05
    assert qp_z > qp_x                                       # stiffer along c


def test_ice_rotated_3d_axis():
    # Skew c-axis (not in any coordinate plane) -> broad off-diagonal coupling.
    axis = np.array([0.5, 0.35, 0.79])
    axis /= np.linalg.norm(axis)
    C6 = an.ice_stiffness_3d(axis)
    off = [abs(C6[0, 3]), abs(C6[0, 4]), abs(C6[1, 3]), abs(C6[2, 5])]
    assert max(off) > 1e8                                    # C14/C15/C24/C36 active
    qp_x = _measure_qp(C6, an.ICE_RHO, axis="x")
    cx = an.christoffel_3d(C6, an.ICE_RHO, (1, 0, 0))[0]
    print(f"3D ice(skew) qP x: {qp_x:.0f} vs Christoffel {cx:.0f} "
          f"({100*(qp_x-cx)/cx:+.1f}%)")
    assert abs(qp_x - cx) / cx < 0.05


def test_footprint_source_and_receive():
    """Finite-aperture transmit/receive: group receive is exactly the weighted
    average of point receives (receive linearity), and multi-point sources run."""
    n = 24
    h = 8.0e-4
    labels = np.zeros((n, n, n), int)                 # single-crystal block
    Cmaps, rho = an.polycrystal_stiffness_3d(labels, [np.array([0.0, 0.0, 1.0])])
    dt = 0.4 * h / (4100.0 * np.sqrt(3))
    nt = 80
    wav = ricker(nt, dt, 0.5e6)
    p1, p2 = (12, 12, 18), (12, 13, 18)
    src_pts = [((12, 12, 4), 0.5), ((12, 13, 4), 0.5)]
    r_pts, _ = elastic3d.forward(Cmaps, rho, h, dt, nt, None, wav, [p1, p2],
                                 src_pts=src_pts)
    r_grp, _ = elastic3d.forward(Cmaps, rho, h, dt, nt, None, wav, None,
                                 src_pts=src_pts,
                                 rec_groups=[([p1, p2], np.array([0.5, 0.5]))])
    assert np.isfinite(r_grp).all()
    assert np.abs(r_grp[:, 0] - 0.5 * (r_pts[:, 0] + r_pts[:, 1])).max() < 1e-15


if __name__ == "__main__":
    test_isotropic_reduction()
    test_ice_on_axis()
    test_ice_rotated_3d_axis()
    test_footprint_source_and_receive()
    print("3D elastic anisotropic checks passed")
