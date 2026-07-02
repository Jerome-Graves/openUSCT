"""Multi-parameter FWI demo: reconstruct sound speed AND attenuation.

A uniform medium contains a separate sound-speed anomaly and attenuation
anomaly. Joint FWI recovers both maps from the same full-matrix-capture data.

Run:  python examples/run_attenuation_demo.py
Output: figures/attenuation.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import attenuation, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def main():
    h = 6.0e-4
    ring = RingArray(n_elements=16, radius_m=0.02, domain_m=0.05, h=h)
    n = ring.n
    dt = 1.0e-7
    nt = 500
    wav = ricker(nt, dt, 0.4e6)

    # True: a fast velocity blob and a separate high-attenuation blob.
    c_true = np.full((n, n), 1500.0)
    c_true = phantom.add_inclusion(c_true, (0.60, 0.50), 0.005, 1800.0, h)
    a_true = np.full((n, n), 0.02)
    a_true = phantom.add_inclusion(a_true, (0.40, 0.55), 0.005, 0.09, h)
    dobs = attenuation.forward_fmc(phantom.velocity_to_m(c_true), a_true, ring, wav, dt, h, nt)

    # Start from a uniform medium; update only inside the ring.
    yy, xx = np.mgrid[0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    mask = (np.hypot(xx - cc, yy - cc) <= 0.02 * 0.9).astype(float)
    m0 = phantom.velocity_to_m(np.full((n, n), 1500.0))
    a0 = np.full((n, n), 0.02)
    m_bounds = (phantom.velocity_to_m(1900.0), phantom.velocity_to_m(1400.0))
    a_bounds = (0.01, 0.12)

    m_rec, a_rec, hist = attenuation.invert(
        m0, a0, ring, wav, dt, h, nt, dobs, n_iter=16,
        step_m=0.03, step_a=0.08, update_mask=mask,
        m_bounds=m_bounds, a_bounds=a_bounds, verbose=True)
    c_rec = phantom.m_to_velocity(m_rec)
    print(f"misfit {hist[0]:.3e} -> {hist[-1]:.3e} ({hist[-1]/hist[0]*100:.1f}%)")

    ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for a_, f, ttl, cmap, vlim in (
        (ax[0, 0], c_true, "True sound speed", "viridis", (1450, 1850)),
        (ax[0, 1], c_rec, "FWI sound speed", "viridis", (1450, 1850)),
        (ax[1, 0], a_true, "True attenuation", "magma", (0.0, 0.1)),
        (ax[1, 1], a_rec, "FWI attenuation", "magma", (0.0, 0.1)),
    ):
        im = a_.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        a_.set_title(ttl); a_.set_xlabel("mm"); fig.colorbar(im, ax=a_, fraction=0.046)
    ax[0, 2].semilogy(np.array(hist) / hist[0], "o-"); ax[0, 2].set_title("Misfit")
    ax[0, 2].set_xlabel("iteration"); ax[0, 2].grid(True, which="both", alpha=0.3)
    ax[1, 2].axis("off")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "attenuation.png")
    fig.savefig(out, dpi=120)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
