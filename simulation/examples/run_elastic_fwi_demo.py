"""Elastic FWI demo: reconstruct a stiffness anomaly from multi-source data.

A homogeneous elastic solid contains a stiffer circular inclusion (faster P and
S). Sources fired around the domain edge, recorded on a surrounding array,
provide the data. Elastic FWI (exact adjoint gradient) recovers the P and S
stiffness maps. This is the elastic counterpart of the acoustic ring-array FWI
and the foundation for crystal-orientation-fabric inversion.

Run:  python examples/run_elastic_fwi_demo.py
Output: figures/elastic_fwi.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import anisotropy as an
from ringfwi.sources import ricker


def ring_indices(n, radius_cells, count):
    cy = cx = n // 2
    idx = []
    for k in range(count):
        a = 2 * np.pi * k / count
        idx.append((int(round(cy + radius_cells * np.sin(a))),
                    int(round(cx + radius_cells * np.cos(a)))))
    return idx


def main():
    n = 64
    h = 4.0e-4
    dt = 4.0e-8
    nt = 260
    wav = ricker(nt, dt, 0.4e6)
    rho = np.full((n, n), 1000.0)

    # True model: homogeneous solid with a stiffer inclusion.
    C11 = np.full((n, n), 9.0e9); C12 = np.full((n, n), 3.0e9)
    C22 = np.full((n, n), 9.0e9); C66 = np.full((n, n), 3.0e9)
    yy, xx = np.mgrid[0:n, 0:n]
    incl = np.hypot(xx - 38, yy - 30) <= 7
    C11t = C11.copy(); C22t = C22.copy(); C66t = C66.copy()
    for Ct in (C11t, C22t):
        Ct[incl] *= 1.4
    C66t[incl] *= 1.4

    src = ring_indices(n, 26, 8)
    rec = ring_indices(n, 28, 40)
    dobs_list = [an._grad_forward(C11t, C12, C22t, C66t, rho, h, dt, nt, s, wav, rec)[0]
                 for s in src]

    # Invert from the homogeneous background inside the ring.
    cc = n // 2
    mask = (np.hypot(xx - cc, yy - cc) <= 26).astype(float)
    bnd = {"C11": (7e9, 15e9), "C22": (7e9, 15e9), "C66": (2e9, 6e9)}
    r11, r12, r22, r66, hist = an.invert(
        C11, C12, C22, C66, rho, h, dt, nt, src, wav, rec, dobs_list,
        n_iter=30, steps={"C11": 0.07, "C22": 0.07, "C66": 0.13},
        update_mask=mask, bounds=bnd, verbose=True)
    print(f"misfit {hist[0]:.3e} -> {hist[-1]:.3e} ({hist[-1]/hist[0]*100:.1f}%)")

    # P and S speeds for display: Vp ~ sqrt(C11/rho), Vs ~ sqrt(C66/rho).
    vp_true = np.sqrt(C11t / rho); vp_rec = np.sqrt(r11 / rho)
    vs_true = np.sqrt(C66t / rho); vs_rec = np.sqrt(r66 / rho)
    ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.4))
    for a_, f, ttl, vlim in (
        (ax[0, 0], vp_true, "True Vp (m/s)", (2900, 3600)),
        (ax[0, 1], vp_rec, "FWI Vp", (2900, 3600)),
        (ax[1, 0], vs_true, "True Vs (m/s)", (1700, 2100)),
        (ax[1, 1], vs_rec, "FWI Vs", (1700, 2100)),
    ):
        im = a_.imshow(f, origin="lower", extent=ext, cmap="viridis",
                       vmin=vlim[0], vmax=vlim[1])
        a_.set_title(ttl); a_.set_xlabel("mm"); fig.colorbar(im, ax=a_, fraction=0.046)
    ax[0, 2].semilogy(np.array(hist) / hist[0], "o-"); ax[0, 2].set_title("Misfit")
    ax[0, 2].set_xlabel("iteration"); ax[0, 2].grid(True, which="both", alpha=0.3)
    ax[1, 2].axis("off")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "elastic_fwi.png")
    fig.savefig(out, dpi=120)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
