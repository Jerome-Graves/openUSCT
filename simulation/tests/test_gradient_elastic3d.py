"""Finite-difference verification of the 3D anisotropic elastic adjoint."""

import numpy as np
import pytest

from ringfwi import anisotropy as an
from ringfwi import elastic3d as e3


@pytest.fixture(scope="module")
def setup():
    n = 10
    labels = np.zeros((n, n, n), int)
    labels[:, :, n // 2:] = 1
    axes = np.array([[0.3, 0.4, 0.87], [0.8, -0.2, 0.57]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    C, rho = an.polycrystal_stiffness_3d(labels, axes)
    C = {k: (np.asarray(v, float).copy() if np.ndim(v)
             else np.full((n, n, n), float(v))) for k, v in C.items()}
    h, dt, nt = 1e-3, 8e-8, 50
    wav = np.zeros(nt)
    wav[:16] = np.sin(np.linspace(0, 2 * np.pi, 16)) * np.hanning(16)
    src = (2, 2, 2)
    rec_idx = [(7, 7, 7), (2, 8, 3)]
    axes2 = axes.copy()
    axes2[1] = [0.7, 0.0, np.sqrt(1 - 0.49)]
    C2, _ = an.polycrystal_stiffness_3d(labels, axes2)
    dobs, _ = e3.forward(C2, rho, h, dt, nt, src, wav, rec_idx)
    return C, rho, h, dt, nt, src, wav, rec_idx, dobs


def test_grad_forward_matches_forward(setup):
    C, rho, h, dt, nt, src, wav, rec_idx, dobs = setup
    r0, _ = e3.forward(C, rho, h, dt, nt, src, wav, rec_idx)
    r1, _ = e3._grad_forward(C, rho, h, dt, nt, src, wav, rec_idx)
    assert np.max(np.abs(r0 - r1)) == 0.0


def test_misfit_zero_at_observed(setup):
    C, rho, h, dt, nt, src, wav, rec_idx, dobs = setup
    r0, _ = e3.forward(C, rho, h, dt, nt, src, wav, rec_idx)
    J, _ = e3.misfit_and_gradient(C, rho, h, dt, nt, src, wav, rec_idx, r0,
                                  grad_keys=("C11",))
    assert J == 0.0


def test_gradient_all_21_keys_fd(setup):
    C, rho, h, dt, nt, src, wav, rec_idx, dobs = setup
    J0, g = e3.misfit_and_gradient(C, rho, h, dt, nt, src, wav, rec_idx, dobs)
    assert J0 > 0
    eps = 1e-5 * float(np.max(np.abs(C["C11"])))
    vox = [(5, 5, 2), (5, 5, 7)]
    for key in e3.GRAD_KEYS:
        for v in vox:
            Cp = {k: val.copy() for k, val in C.items()}
            Cp[key][v] += eps
            rp, _ = e3.forward(Cp, rho, h, dt, nt, src, wav, rec_idx)
            Cm = {k: val.copy() for k, val in C.items()}
            Cm[key][v] -= eps
            rm, _ = e3.forward(Cm, rho, h, dt, nt, src, wav, rec_idx)
            fd = (0.5 * np.sum((rp - dobs) ** 2)
                  - 0.5 * np.sum((rm - dobs) ** 2)) / (2 * eps)
            ad = g[key][v]
            if abs(fd) + abs(ad) < 1e-16:
                continue
            rel = abs(fd - ad) / (abs(fd) + abs(ad))
            assert rel < 1e-5, f"{key} @ {v}: FD={fd:e} ADJ={ad:e} rel={rel:e}"
