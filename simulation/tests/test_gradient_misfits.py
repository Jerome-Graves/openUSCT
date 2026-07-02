"""Finite-difference gradient checks for the robust misfit functionals.

Verifies the adjoint sources of the envelope, envelope-GCN, and
cross-correlation traveltime misfits against a central finite-difference
directional derivative of the full FWI gradient. The envelope adjoints are
exact (the FFT Hilbert transpose is -H), so they should match to FD-truncation
precision; the traveltime adjoint is the classic Luo & Schuster stationary-
phase result and is exact only to the stationarity approximation, so it gets a
looser tolerance.
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
    nt = 350
    wav = ricker(nt, dt, 0.4e6)
    c_true = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c_true = phantom.add_inclusion(c_true, (0.5, 0.5), 0.006, 3400.0, h)
    m_true = phantom.velocity_to_m(c_true)
    m0 = phantom.velocity_to_m(
        phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h))
    src_list = [0, 4]
    dobs = fwi.forward_fmc(m_true, geom, wav, dt, h, nt, src_list=src_list)
    return geom, wav, dt, h, nt, dobs, src_list, m0, n


def _directional_check(misfit_type, eps_frac, tol):
    geom, wav, dt, h, nt, dobs, src_list, m0, n = _setup()
    rng = np.random.default_rng(1)

    J0, g = fwi.misfit_and_gradient(m0, geom, wav, dt, h, nt, dobs,
                                    src_list=src_list, misfit_type=misfit_type)
    # random direction, supported away from the boundary
    dm = rng.standard_normal((n, n))
    dm[:5, :] = dm[-5:, :] = dm[:, :5] = dm[:, -5:] = 0.0
    dm *= float(np.mean(m0))
    analytic = float(np.sum(g * dm))

    eps = eps_frac
    Jp = fwi.misfit(m0 + eps * dm, geom, wav, dt, h, nt, dobs,
                    src_list=src_list, misfit_type=misfit_type)
    Jm = fwi.misfit(m0 - eps * dm, geom, wav, dt, h, nt, dobs,
                    src_list=src_list, misfit_type=misfit_type)
    fd = (Jp - Jm) / (2.0 * eps)
    rel = abs(analytic - fd) / (abs(fd) + 1e-30)
    print(f"{misfit_type}: J={J0:.4e} analytic={analytic:.6e} fd={fd:.6e} "
          f"rel={rel:.2e}")
    assert rel < tol, f"{misfit_type} gradient mismatch: rel={rel}"


def test_envelope_gradient():
    _directional_check("envelope", 1.0e-6, 1e-3)


def test_egcn_gradient():
    _directional_check("egcn", 1.0e-6, 1e-3)


def test_traveltime_gradient():
    # Stationary-phase adjoint: agreement to ~10% validates sign and scale.
    _directional_check("traveltime", 1.0e-5, 0.15)


def test_gsot_gradient():
    # Exact once the assignment is fixed; a small FD step keeps it fixed.
    _directional_check("gsot", 1.0e-7, 1e-2)


if __name__ == "__main__":
    test_envelope_gradient()
    test_egcn_gradient()
    test_traveltime_gradient()
    print("robust-misfit gradient checks passed")
