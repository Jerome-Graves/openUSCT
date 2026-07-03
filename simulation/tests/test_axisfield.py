"""Tests for the 3D voxel-wise orientation-field machinery."""

import numpy as np
import pytest

from ringfwi import anisotropy as an
from ringfwi import axisfield as af
from ringfwi import elastic3d as e3
from ringfwi.cof import axes_from_params


def test_bond_rotation_matches_tensor_rotation():
    base6 = an.ti_stiffness_6(**an.ICE_MATERIAL)
    rng = np.random.default_rng(1)
    t = rng.uniform(0, np.pi / 2, 4)
    p = rng.uniform(-np.pi, np.pi, 4)
    C, _, _ = af.axis_stiffness_and_derivs(t, p, base6)
    for k in range(4):
        ax = axes_from_params([[t[k], p[k]]])[0]
        ref = an.ti_stiffness_3d(ax, an.ICE_MATERIAL)
        assert np.max(np.abs(C[k] - ref)) / np.max(np.abs(ref)) < 1e-12


def test_stiffness_angle_derivatives_fd():
    base6 = an.ti_stiffness_6(**an.ICE_MATERIAL)
    t = np.array([0.4, 1.1])
    p = np.array([0.7, -2.0])
    _, dCt, dCp = af.axis_stiffness_and_derivs(t, p, base6)
    eps = 1e-6
    Cp_, _, _ = af.axis_stiffness_and_derivs(t + eps, p, base6)
    Cm_, _, _ = af.axis_stiffness_and_derivs(t - eps, p, base6)
    assert np.max(np.abs((Cp_ - Cm_) / (2 * eps) - dCt)) \
        / np.max(np.abs(dCt)) < 1e-6
    Cp_, _, _ = af.axis_stiffness_and_derivs(t, p + eps, base6)
    Cm_, _, _ = af.axis_stiffness_and_derivs(t, p - eps, base6)
    assert np.max(np.abs((Cp_ - Cm_) / (2 * eps) - dCp)) \
        / np.max(np.abs(dCp)) < 1e-6


def test_block_partition_covers_mask():
    mask = np.zeros((12, 12, 12), bool)
    mask[2:10, 3:11, 1:9] = True
    labels, nb = af.block_partition(mask, 2)
    assert nb == 8
    assert (labels[mask] >= 0).all()
    assert (labels[~mask] == -1).all()
    assert set(np.unique(labels[mask])) == set(range(8))


def test_smooth_axes_respects_c_symmetry():
    # a field mixing a and -a voxel to voxel is one constant orientation
    mask = np.ones((8, 8, 8), bool)
    colat = np.full(mask.shape, 2.0)          # points into lower hemisphere
    colat[::2] = np.pi - 2.0                  # the antipodal axis
    azim = np.full(mask.shape, 0.5)
    azim[::2] = 0.5 - np.pi
    sc, sa = af.smooth_axes(colat, azim, mask, sigma=1.5)
    a = af.field_to_axes(sc, sa)
    ref = af.field_to_axes(np.full((1,), np.pi - 2.0),
                           np.full((1,), 0.5 - np.pi))[0]
    dots = np.abs(a.reshape(-1, 3) @ ref)
    assert dots.min() > 1 - 1e-9


@pytest.fixture(scope="module")
def small_problem():
    n = 12
    mask = np.zeros((n, n, n), bool)
    mask[3:9, 3:9, 3:9] = True
    mat = an.ICE_MATERIAL
    base6 = an.ti_stiffness_6(**mat)
    K = 1000.0 * 1480.0 ** 2
    Cbg = {}
    for key in af.KEYS21:
        i, j = int(key[1]) - 1, int(key[2]) - 1
        Cbg[key] = np.full((n, n, n), K if (i < 3 and j < 3) else 0.0)
    h, dt, nt = 1e-3, 8e-8, 50
    wav = np.zeros(nt)
    wav[:16] = np.sin(np.linspace(0, 2 * np.pi, 16)) * np.hanning(16)
    src_pts_list = [[((1, 6, 6), 1.0)], [((10, 5, 5), 1.0)]]
    rec_groups = [([(6, 1, 6)], [1.0]), ([(6, 10, 6)], [1.0]),
                  ([(1, 1, 6)], [1.0])]
    colat_true = np.where(mask, 0.9, 0.0)
    azim_true = np.where(mask, -0.4, 0.0)
    Ct, rho_t = af.build_maps(colat_true, azim_true, mask, base6, Cbg,
                              1000.0, mat["rho"])
    dobs = np.zeros((2, nt, 3))
    for i, sp in enumerate(src_pts_list):
        dobs[i], _ = e3.forward(Ct, rho_t, h, dt, nt, None, wav, None,
                                src_pts=sp, rec_groups=rec_groups)
    return (mask, base6, Cbg, mat, h, dt, nt, wav, src_pts_list, rec_groups,
            colat_true, azim_true, dobs)


def test_orientation_gradient_fd(small_problem):
    (mask, base6, Cbg, mat, h, dt, nt, wav, src_pts_list, rec_groups,
     colat_true, azim_true, dobs) = small_problem
    rng = np.random.default_rng(2)
    colat = np.where(mask, 0.5, 0.0) + 0.1 * rng.standard_normal(mask.shape) * mask
    azim = np.where(mask, 0.8, 0.0) + 0.2 * rng.standard_normal(mask.shape) * mask
    J0, g_t, g_p = af.misfit_and_gradient_axes(
        colat, azim, mask, base6, Cbg, 1000.0, mat["rho"], h, dt, nt,
        src_pts_list, wav, rec_groups, dobs)
    assert J0 > 0

    def Jof(cl, az):
        Cm, rho = af.build_maps(cl, az, mask, base6, Cbg, 1000.0, mat["rho"])
        Jtot = 0.0
        for i, sp in enumerate(src_pts_list):
            r, _ = e3.forward(Cm, rho, h, dt, nt, None, wav, None,
                              src_pts=sp, rec_groups=rec_groups)
            Jtot += 0.5 * np.sum((r - dobs[i]) ** 2)
        return Jtot

    eps = 1e-6
    for v in [(4, 4, 4), (7, 6, 5)]:
        for which, gmap in (("colat", g_t), ("azim", g_p)):
            cp = colat.copy(); ap = azim.copy()
            cm = colat.copy(); am = azim.copy()
            if which == "colat":
                cp[v] += eps; cm[v] -= eps
            else:
                ap[v] += eps; am[v] -= eps
            fd = (Jof(cp, ap) - Jof(cm, am)) / (2 * eps)
            ad = gmap[v]
            rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-30)
            assert rel < 1e-5, f"{which} @ {v}: FD={fd:e} ADJ={ad:e}"


def test_gradient_descent_reduces_misfit_and_error(small_problem):
    (mask, base6, Cbg, mat, h, dt, nt, wav, src_pts_list, rec_groups,
     colat_true, azim_true, dobs) = small_problem
    # start inside the local basin (~10 degrees off truth)
    colat = np.where(mask, 0.9 + 0.17, 0.0)
    azim = np.where(mask, -0.4 - 0.17, 0.0)
    true_ax = af.field_to_axes(colat_true, azim_true)
    err0 = af.axis_error_map(colat, azim, true_ax, mask)[mask].mean()
    hist = []
    for it in range(8):
        J, g_t, g_p = af.misfit_and_gradient_axes(
            colat, azim, mask, base6, Cbg, 1000.0, mat["rho"], h, dt, nt,
            src_pts_list, wav, rec_groups, dobs)
        hist.append(J)
        colat, azim = af.gradient_step(colat, azim, mask, g_t, g_p,
                                       step_rad=0.3, smooth_sigma=1.0)
    err1 = af.axis_error_map(colat, azim, true_ax, mask)[mask].mean()
    assert hist[-1] < 0.3 * hist[0]
    # 2 sources x 3 receivers is heavily underdetermined, so the voxel-wise
    # error need not fall as fast as the misfit -- but it must not blow up.
    assert err1 < err0 + 1.0


def test_field_misfit_matches_gradient_misfit(small_problem):
    (mask, base6, Cbg, mat, h, dt, nt, wav, src_pts_list, rec_groups,
     colat_true, azim_true, dobs) = small_problem
    colat = np.where(mask, 0.6, 0.0)
    azim = np.where(mask, 0.3, 0.0)
    tw = np.ones((nt, 3)); tw[:, 1] = 0.5
    J_fwd = af.field_misfit(colat, azim, mask, base6, Cbg, 1000.0,
                            mat["rho"], h, dt, nt, src_pts_list, wav,
                            rec_groups, dobs, trace_weights=tw)
    J_adj, _, _ = af.misfit_and_gradient_axes(colat, azim, mask, base6, Cbg,
                                              1000.0, mat["rho"], h, dt, nt,
                                              src_pts_list, wav, rec_groups,
                                              dobs, trace_weights=tw)
    assert abs(J_fwd - J_adj) / J_adj < 1e-12


def test_segment_field_finds_two_regions():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:8, 2:8, 2:8] = True
    colat = np.zeros(mask.shape); azim = np.zeros(mask.shape)
    left = mask & (np.arange(10)[None, None, :] < 5)
    right = mask & ~left
    colat[left] = 0.2; azim[left] = 0.1
    colat[right] = 1.3; azim[right] = -2.0
    seg, sizes, order = af.segment_field(colat, azim, mask, n_clusters=4,
                                         seed=1)
    assert (seg[~mask] == -1).all()
    # the two largest segments together cover the mask and split it cleanly
    a, b = order[0], order[1]
    assert sizes[a] + sizes[b] == int(mask.sum())
    for region in (left, right):
        ids = np.unique(seg[region])
        assert len(ids) == 1


def test_set_segment_axis():
    mask = np.ones((6, 6, 6), bool)
    seg = np.zeros(mask.shape, int); seg[:, :, 3:] = 1
    colat = np.full(mask.shape, 0.7); azim = np.full(mask.shape, 0.2)
    c2, a2 = af.set_segment_axis(colat, azim, seg, 1, [0.0, 0.0, -1.0])
    assert np.allclose(c2[seg == 1], 0.0)          # -c flipped to +c
    assert np.allclose(c2[seg == 0], 0.7)
    assert np.allclose(a2[seg == 0], 0.2)
