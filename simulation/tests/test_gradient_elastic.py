"""Finite-difference check of the elastic FWI gradient.

The exact discrete adjoint gradient of the data misfit with respect to the
stiffness maps (C11, C12, C22, C66) is compared to a central finite-difference
directional derivative. Reverse-mode differentiation of the exact discrete
forward should match to near machine precision.
"""

from __future__ import annotations

import numpy as np

from ringfwi import anisotropy as an
from ringfwi.sources import ricker


def test_elastic_gradient():
    rng = np.random.default_rng(0)
    n = 44
    h = 4.0e-4
    dt = 4.0e-8
    nt = 240
    wav = ricker(nt, dt, 0.4e6)

    # Isotropic-ish background so all four stiffnesses are exercised.
    rho = np.full((n, n), 1000.0)
    base = dict(C11=9.0e9, C12=3.0e9, C22=9.5e9, C66=3.2e9)
    C11 = np.full((n, n), base["C11"]); C12 = np.full((n, n), base["C12"])
    C22 = np.full((n, n), base["C22"]); C66 = np.full((n, n), base["C66"])

    src = (n // 2, n // 3)
    rec = [(10, 30), (33, 12), (30, 33), (12, 20)]

    # Observed data from a perturbed "true" model.
    Ct = [C11 * 1.03, C12 * 0.97, C22 * 1.02, C66 * 1.05]
    dobs, _, _ = an._grad_forward(*Ct, rho, h, dt, nt, src, wav, rec)

    J, g = an.misfit_and_gradient(C11, C12, C22, C66, rho, h, dt, nt,
                                  src, wav, rec, dobs)

    # Random perturbation direction (scaled to each stiffness magnitude).
    dC = {k: rng.standard_normal((n, n)) * base[k] for k in base}
    analytic = sum(float(np.sum(g[k] * dC[k])) for k in base)

    eps = 1.0e-4       # central difference: rel error scales as O(eps^2)
    def perturb(sgn):
        Cs = [np.full((n, n), 0.0) for _ in range(4)]
        for i, k in enumerate(("C11", "C12", "C22", "C66")):
            C0 = {"C11": C11, "C12": C12, "C22": C22, "C66": C66}[k]
            Cs[i] = C0 + sgn * eps * dC[k]
        d, _, _ = an._grad_forward(*Cs, rho, h, dt, nt, src, wav, rec)
        return 0.5 * float(np.sum((d - dobs) ** 2))

    fd = (perturb(+1) - perturb(-1)) / (2 * eps)
    rel = abs(analytic - fd) / abs(fd)
    print(f"J={J:.6e}  analytic={analytic:.6e}  fd={fd:.6e}  rel_err={rel:.2e}")
    assert rel < 1e-5


if __name__ == "__main__":
    test_elastic_gradient()
    print("elastic gradient check passed")
