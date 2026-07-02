"""End-to-end 3D ring-array FWI demonstration.

A cylindrical specimen (sound speed 3000 m/s) in a water bath contains a small
spherical low-velocity flaw. A cylindrical array (rings stacked along the axis)
records a full-matrix-capture dataset. Starting from the correct background but
without the flaw, 3D adjoint-state FWI recovers the flaw as a localised volume.

This is compute-heavy (3D forward and adjoint solves), so it runs on a modest
grid and takes a few minutes on a laptop CPU.

Run:  python examples/run_fwi_demo_3d.py
Output: figures/reconstruction_3d.png
"""

from __future__ import annotations

import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import fwi, phantom
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


def main():
    # ---- Acquisition and grid -------------------------------------------------
    h = 1.25e-3
    domain = 0.036
    geom = CylinderArray(n_rings=3, per_ring=12, radius_m=0.015,
                         height_m=0.014, domain_m=domain, h=h)
    n = geom.n

    dt = 1.2e-7
    nt = 300
    wavelet = ricker(nt, dt, 0.3e6)

    c_specimen, c_couplant = 3000.0, 1480.0
    spec_radius = 0.012

    # ---- True model -----------------------------------------------------------
    c_true = phantom.cylinder_background((n, n, n), c_specimen, c_couplant, spec_radius, h)
    c_true = phantom.add_sphere(c_true, (0.56, 0.50, 0.50), 0.005, 2650.0, h)
    m_true = phantom.velocity_to_m(c_true)

    # ---- Starting model: correct background, no flaw --------------------------
    c0 = phantom.cylinder_background((n, n, n), c_specimen, c_couplant, spec_radius, h)
    m0 = phantom.velocity_to_m(c0)

    # Update only inside the specimen.
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
    cxy = (n - 1) * h / 2.0
    r_xy = np.hypot(xx - cxy, yy - cxy)
    update_mask = (r_xy <= spec_radius * 0.95).astype(float)

    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2500.0))

    # Transmit from a subset of elements to keep runtime modest; record on all.
    src_list = list(range(0, geom.n_elements, 3))
    print(f"grid {n}x{n}x{n}, {geom.n_elements} elements, {len(src_list)} transmitters, nt={nt}")

    # ---- Observed data --------------------------------------------------------
    t0 = time.time()
    dobs = fwi.forward_fmc(m_true, geom, wavelet, dt, h, nt, src_list=src_list)
    print(f"forward FMC simulated in {time.time() - t0:.1f} s")

    # ---- Invert ---------------------------------------------------------------
    t0 = time.time()
    m_rec, history = fwi.invert(
        m0, geom, wavelet, dt, h, nt, dobs, src_list=src_list,
        n_iter=8, step_frac=0.04,
        update_mask=update_mask, m_bounds=m_bounds, smooth_sigma=1.0, verbose=True,
    )
    print(f"inversion in {time.time() - t0:.1f} s")

    c_rec = phantom.m_to_velocity(m_rec)
    c_tru = phantom.m_to_velocity(m_true)

    # ---- Figure: three orthogonal slices through the flaw ---------------------
    iz = int(round(0.50 * (n - 1)))
    iy = int(round(0.50 * (n - 1)))
    ix = int(round(0.56 * (n - 1)))
    vmin, vmax = 2600, 3100
    ext = [0, domain * 1e3, 0, domain * 1e3]

    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
    slices = [
        ("xy (axial)", lambda v: v[iz], ext),
        ("xz", lambda v: v[:, iy, :], ext),
        ("yz", lambda v: v[:, :, ix], ext),
    ]
    for col, (name, sl, e) in enumerate(slices):
        for row, (vol, tag) in enumerate(((c_tru, "true"), (c_rec, "FWI"))):
            a = ax[row, col]
            im = a.imshow(sl(vol), origin="lower", extent=e, cmap="viridis", vmin=vmin, vmax=vmax)
            a.set_title(f"{tag}: {name} slice")
            a.set_xlabel("mm"); a.set_ylabel("mm")
            fig.colorbar(im, ax=a, fraction=0.046, label="c (m/s)")

    ax[0, 3].semilogy(np.array(history) / history[0], "o-")
    ax[0, 3].set_title("Misfit convergence")
    ax[0, 3].set_xlabel("iteration"); ax[0, 3].set_ylabel("normalised misfit")
    ax[0, 3].grid(True, which="both", alpha=0.3)
    ax[1, 3].axis("off")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "reconstruction_3d.png")
    fig.savefig(out, dpi=120)
    print(f"saved {os.path.normpath(out)}")
    print(f"misfit reduced to {history[-1] / history[0] * 100:.1f}% of initial")


if __name__ == "__main__":
    main()
