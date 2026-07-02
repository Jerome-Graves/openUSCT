"""Elastic wavefield demo: P/S propagation and mode conversion.

An explosive (P) source in a solid launches a fast compressional front. When it
strikes a stiffer inclusion, part of the energy converts to a slower shear (S)
wave. The velocity-magnitude snapshots show both fronts and the converted S
energy that a purely acoustic model cannot represent. This mode conversion is
exactly the signal that makes elastic imaging valuable for grain-boundary and
crystal-orientation-fabric work.

Run:  python examples/run_elastic_demo.py
Output: figures/elastic.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import elastic
from ringfwi.sources import ricker


def main():
    n = 220
    h = 1.0e-3
    dt = 1.2e-7
    nt = 260
    wav = ricker(nt, dt, 0.25e6)

    # Background solid with a stiffer, faster circular inclusion.
    vp = np.full((n, n), 3000.0)
    vs = np.full((n, n), 1500.0)
    rho = np.full((n, n), 1000.0)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    incl = np.hypot(xx - 150, yy - 110) <= 28
    vp[incl] = 4500.0
    vs[incl] = 2600.0
    rho[incl] = 1300.0

    src = (110, 70)
    _, hist = elastic.forward(vp, vs, rho, h, dt, nt, src, wav, [src],
                              source="explosive", record="pressure", store=True)

    snaps = [90, 150, 220]
    ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for a_, k in zip(ax, snaps):
        frame = hist[k]
        vmax = 0.4 * frame.max() + 1e-30
        a_.imshow(frame, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vmax)
        th = np.linspace(0, 2 * np.pi, 200)
        a_.plot(150 * h * 1e3 + 28 * h * 1e3 * np.cos(th),
                110 * h * 1e3 + 28 * h * 1e3 * np.sin(th),
                "c--", lw=1.0, alpha=0.7)
        a_.plot(src[1] * h * 1e3, src[0] * h * 1e3, "w*", ms=10)
        a_.set_title(f"|v| at t = {k * dt * 1e6:.1f} us")
        a_.set_xlabel("mm"); a_.set_ylabel("mm")
    fig.suptitle("Elastic wavefield: fast P front, slow S front, and P->S "
                 "conversion at the inclusion (dashed)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "elastic.png")
    fig.savefig(out, dpi=120)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
