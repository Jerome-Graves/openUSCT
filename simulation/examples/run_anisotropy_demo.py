"""Anisotropy demo: directional P-wave velocity in single-crystal ice.

Ice Ih is hexagonal: its P-wave speed depends on the propagation direction
relative to the c-axis. The left panel shows the qP phase-velocity surface from
the Christoffel equation for two c-axis orientations. The right panel shows a
point-source wavefield in homogeneous ice: the qP front is visibly non-circular,
faster along the c-axis. This directional velocity anisotropy is what
crystal-orientation-fabric (COF) estimation measures.

Run:  python examples/run_anisotropy_demo.py
Output: figures/anisotropy.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import anisotropy as an
from ringfwi.sources import ricker


def main():
    # --- qP velocity surface (Christoffel) ---
    phis = np.linspace(0, 2 * np.pi, 361)
    surfaces = {}
    for th_deg in (0, 45):
        C = an.ice_stiffness_2d(np.radians(th_deg))
        surfaces[th_deg] = np.array(
            [an.christoffel_velocities(C, an.ICE_RHO, p)[0] for p in phis])

    # --- point-source wavefield in homogeneous ice (c-axis along y) ---
    n = 161
    h = 3.0e-4
    dt = 3.0e-8
    nt = 300
    wav = ricker(nt, dt, 0.45e6)
    C = an.ice_stiffness_2d(0.0)
    Cmaps = (C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2])
    rho = np.full((n, n), an.ICE_RHO)
    src = (n // 2, n // 2)
    _, hist = an.forward(Cmaps, rho, h, dt, nt, src, wav, [src],
                         source="explosive", record="pressure", store=True)
    ksnap = 165
    snap = hist[ksnap]

    fig = plt.figure(figsize=(13, 5.6))
    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    for th_deg, v in surfaces.items():
        ax0.plot(phis, v, lw=2, label=f"c-axis at {th_deg} deg")
    ax0.set_title("qP phase velocity in ice Ih (m/s)\nChristoffel equation", pad=18)
    ax0.set_rlim(3600, 4150)
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    ax1 = fig.add_subplot(1, 2, 2)
    ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
    ax1.imshow(snap, origin="lower", extent=ext, cmap="inferno",
               vmin=0, vmax=0.4 * snap.max() + 1e-30)
    c = (n // 2) * h * 1e3
    ax1.annotate("", xy=(c, c + 12), xytext=(c, c - 12),
                 arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.5))
    ax1.text(c + 1.5, c + 12, "c-axis", color="cyan", fontsize=10)
    ax1.set_title(f"qP wavefront at t = {ksnap * dt * 1e6:.1f} us\n"
                  "(faster along the c-axis -> non-circular)")
    ax1.set_xlabel("mm"); ax1.set_ylabel("mm")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "anisotropy.png")
    fig.savefig(out, dpi=120)
    v0, v90 = surfaces[0][0], surfaces[0][90]
    print(f"ice qP: {v0:.0f} m/s across c-axis, {v90:.0f} m/s along c-axis "
          f"({100*(v90-v0)/v0:+.1f}% anisotropy)")
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
