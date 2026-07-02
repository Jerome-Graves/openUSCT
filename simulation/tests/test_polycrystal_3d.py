"""Verify the 3D Voronoi polycrystal and its reduced (apparent-speed) model."""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import fwi, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import gabor


def test_tessellation_fills_cylinder():
    rng = np.random.default_rng(5)
    n = 36
    h = 7.0e-4
    radius = 0.009
    labels, axes, colat = phantom.voronoi_polycrystal_3d((n, n, n), 10, radius,
                                                         h, rng=rng, relax=1)
    c = (n - 1) * h / 2.0
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
    inside = np.hypot(x - c, y - c) <= radius
    assert (labels[inside] >= 0).all()
    assert (labels[~inside] == -1).all()
    assert len(np.unique(labels[labels >= 0])) >= 8
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)
    assert (axes[:, 2] >= 0).all()                   # upper hemisphere
    assert (colat[inside] >= 0).all() and (colat[inside] <= np.pi / 2 + 1e-9).all()


def test_qp_vs_caxis_limits():
    # Along the c-axis (psi=0): sqrt(C33/rho); across it (psi=90): sqrt(C11/rho).
    v_par = an.ice_qp_vs_caxis(0.0)
    v_perp = an.ice_qp_vs_caxis(np.pi / 2.0)
    assert abs(v_par - np.sqrt(an.ICE_C33 / an.ICE_RHO)) < 1.0
    assert abs(v_perp - np.sqrt(an.ICE_C11 / an.ICE_RHO)) < 1.0
    # Intermediate angles dip below both (the TI qP surface minimum).
    v_mid = an.ice_qp_vs_caxis(np.radians(45.0))
    assert v_mid < min(v_par, v_perp)


def test_apparent_speed_and_forward():
    rng = np.random.default_rng(5)
    n = 36
    h = 7.0e-4
    radius = 0.009
    labels, axes, colat = phantom.voronoi_polycrystal_3d((n, n, n), 10, radius,
                                                         h, rng=rng, relax=1)
    capp = an.polycrystal_apparent_speed_3d(labels, axes)
    ice = labels >= 0
    assert 3750.0 < capp[ice].min() and capp[ice].max() < 4050.0
    assert capp[ice].max() - capp[ice].min() > 20.0   # real orientation contrast
    assert np.allclose(capp[~ice], 1480.0)

    # Acoustic forward through the reduced model runs and is finite.
    d = (n - 1) * h
    cyl = CylinderArray(n_rings=2, per_ring=6, radius_m=0.011, height_m=0.010,
                        domain_m=d, h=h)
    dt = 0.5 * h / (capp.max() * np.sqrt(3))
    nt = 160
    wav = gabor(nt, dt, 0.4e6, 0.6)
    data = fwi.forward_fmc(phantom.velocity_to_m(capp), cyl, wav, dt, h, nt,
                           src_list=[0])
    assert np.isfinite(data).all() and np.abs(data).max() > 0


def test_material_presets():
    """Every TI preset is physically admissible and Christoffel-consistent."""
    for name, mat in an.TI_MATERIALS.items():
        C6 = an.ti_stiffness_6(**mat)
        assert np.linalg.eigvalsh(C6).min() > 0, f"{name} not positive definite"
        qp_z = an.christoffel_3d(C6, mat["rho"], (0, 0, 1))[0]
        qp_x = an.christoffel_3d(C6, mat["rho"], (1, 0, 0))[0]
        assert abs(qp_z - np.sqrt(mat["C33"] / mat["rho"])) < 1.0
        assert abs(qp_x - np.sqrt(mat["C11"] / mat["rho"])) < 1.0
        assert an.ti_max_speed(mat) >= max(qp_z, qp_x) - 1.0


def test_polycrystal_other_material():
    """Titanium polycrystal maps carry the right density and speed range."""
    rng = np.random.default_rng(5)
    labels, axes, _ = phantom.voronoi_polycrystal_3d((30, 30, 30), 8, 0.008,
                                                     8.0e-4, rng=rng)
    ti = an.TI_MATERIALS["Titanium (alpha)"]
    Cmaps, rho = an.polycrystal_stiffness_3d(labels, axes, material=ti)
    ice_cells = labels >= 0
    assert np.allclose(rho[ice_cells], ti["rho"])
    assert np.allclose(rho[~ice_cells], 1000.0)
    capp = an.polycrystal_apparent_speed_3d(labels, axes, material=ti)
    v_lo = np.sqrt(min(ti["C11"], ti["C33"]) / ti["rho"]) * 0.9
    v_hi = an.ti_max_speed(ti) * 1.001
    assert v_lo < capp[ice_cells].min() and capp[ice_cells].max() < v_hi


if __name__ == "__main__":
    test_tessellation_fills_cylinder()
    test_qp_vs_caxis_limits()
    test_apparent_speed_and_forward()
    test_material_presets()
    test_polycrystal_other_material()
    print("3D polycrystal checks passed")
