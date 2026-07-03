"""Tests for the Voronoi-seed (parametric unknown-geometry) inversion."""

import numpy as np

from ringfwi import anisotropy as an
from ringfwi import voronoi_inv as vi


def _mask(n=12):
    m = np.zeros((n, n, n), bool)
    m[2:10, 2:10, 2:10] = True
    return m


def test_params_roundtrip():
    seeds = np.array([[0.01, 0.02, 0.003], [0.004, 0.005, 0.006]])
    axes = np.array([[0.3, 0.4, 0.87], [0.0, 0.0, 1.0]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    s2, a2 = vi.params_unpack(vi.params_pack(seeds, axes))
    assert np.allclose(s2, seeds)
    assert np.allclose(np.abs(np.sum(a2 * axes, axis=1)), 1.0, atol=1e-12)


def test_soft_weights_hard_limit():
    mask = _mask()
    h = 1e-3
    seeds = np.array([[0.003, 0.005, 0.004], [0.008, 0.005, 0.006]])
    lab = vi.hard_labels(seeds, mask, h)
    w = vi.soft_weights(seeds, mask, h, tau_voxels=0.1)   # near-hard
    assert np.allclose(w.sum(axis=1), 1.0)
    hard_from_soft = np.argmax(w, axis=1)
    assert (hard_from_soft == lab[mask]).mean() > 0.99


def test_blended_maps_single_grain_matches_rotation():
    mask = _mask()
    h = 1e-3
    mat = an.ICE_MATERIAL
    base6 = an.ti_stiffness_6(**mat)
    K = 1000.0 * 1480.0 ** 2
    Cbg = {k: np.full(mask.shape,
                      K if (int(k[1]) <= 3 and int(k[2]) <= 3) else 0.0)
           for k in vi._KEYS21}
    ax = np.array([[0.3, 0.4, 0.87]])
    ax = ax / np.linalg.norm(ax)
    seeds = np.array([[0.005, 0.005, 0.005]])
    Cm, rho = vi.blended_maps(seeds, ax, mask, h, base6, Cbg,
                              1000.0, mat["rho"])
    ref = an.ti_stiffness_3d(ax[0], mat)
    for key in ("C11", "C14", "C56", "C36"):
        i, j = int(key[1]) - 1, int(key[2]) - 1
        got = Cm[key][mask]
        assert np.allclose(got, ref[i, j], rtol=1e-10, atol=1e-2), key
    assert np.allclose(rho[mask], mat["rho"])
    assert np.allclose(rho[~mask], 1000.0)


def test_kmeans_seeds_inside_specimen():
    mask = _mask()
    h = 1e-3
    seeds = vi.kmeans_seeds(mask, h, 4, seed=1)
    assert seeds.shape == (4, 3)
    lab = vi.hard_labels(seeds, mask, h)
    assert set(np.unique(lab[mask])) == set(range(4))


def test_fd_steps_layout():
    h = 1e-3
    fd = vi.fd_steps(2, h)
    assert fd.shape == (10,)
    assert np.allclose(fd[:3], h)
    assert np.allclose(fd[3:5], np.radians(2.0))
    assert np.allclose(fd[5:8], h)
