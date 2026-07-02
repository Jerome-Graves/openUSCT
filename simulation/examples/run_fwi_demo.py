"""End-to-end ring-array FWI demonstration.

A circular specimen (sound speed 3000 m/s, representative of ice) sits in a
water couplant bath and contains a small higher-velocity inclusion (a "flaw").
A ring of transducers records a full-matrix-capture dataset. Starting from the
correct background but *without* the inclusion, adjoint-state FWI recovers the
inclusion from the waveform data alone.

Run:  python examples/run_fwi_demo.py
Output: figures/reconstruction.png
"""

from __future__ import annotations

import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.solver import make_sponge
from ringfwi.sources import ricker


def main():
    # ---- Acquisition and grid -------------------------------------------------
    h = 6.0e-4                  # 0.6 mm grid
    domain = 0.06               # 60 mm square
    geom = RingArray(n_elements=16, radius_m=0.024, domain_m=domain, h=h)
    n = geom.n

    dt = 7.0e-8
    nt = 800
    f0 = 0.25e6                 # 250 kHz Ricker
    wavelet = ricker(nt, dt, f0)
    # The exact discrete adjoint is used (sponge=None), so the box boundary is a
    # consistent part of the forward model shared by observed and synthetic data.
    sponge = None

    c_specimen, c_couplant = 3000.0, 1480.0
    spec_radius = 0.020

    # ---- True model: specimen + couplant + inclusion --------------------------
    c_true = phantom.coupling_background((n, n), c_specimen, c_couplant, spec_radius, h)
    c_true = phantom.add_inclusion(c_true, (0.58, 0.46), 0.007, 2650.0, h)
    m_true = phantom.velocity_to_m(c_true)

    # ---- Starting model: correct background, no inclusion ---------------------
    c_start = phantom.coupling_background((n, n), c_specimen, c_couplant, spec_radius, h)
    m0 = phantom.velocity_to_m(c_start)

    # Only allow updates inside the specimen disc.
    yy, xx = np.mgrid[0:n, 0:n] * h
    centre = (n - 1) * h / 2.0
    r = np.hypot(xx - centre, yy - centre)
    update_mask = (r <= spec_radius * 0.95).astype(float)

    m_bounds = (phantom.velocity_to_m(4200.0), phantom.velocity_to_m(2600.0))

    print(f"grid {n}x{n}, {geom.n_elements} elements, nt={nt}")

    # ---- Observed data --------------------------------------------------------
    t0 = time.time()
    dobs = fwi.forward_fmc(m_true, geom, wavelet, dt, h, nt, sponge)
    print(f"forward FMC simulated in {time.time() - t0:.1f} s")

    # ---- Invert ---------------------------------------------------------------
    t0 = time.time()
    m_rec, history = fwi.invert(
        m0, geom, wavelet, dt, h, nt, dobs, sponge,
        n_iter=12, step_frac=0.03,
        update_mask=update_mask, m_bounds=m_bounds, verbose=True,
    )
    print(f"inversion ({len(history)} iters) in {time.time() - t0:.1f} s")

    c_rec = phantom.m_to_velocity(m_rec)

    # ---- Figure ---------------------------------------------------------------
    vmin, vmax = 2600, 3100
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ext = [0, domain * 1e3, 0, domain * 1e3]

    for a, field, title in (
        (ax[0], phantom.m_to_velocity(m_true), "True model"),
        (ax[1], phantom.m_to_velocity(m0), "Starting model"),
        (ax[2], c_rec, "FWI reconstruction"),
    ):
        im = a.imshow(field, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
        a.plot(geom.xy[:, 0] * 1e3, geom.xy[:, 1] * 1e3, "w.", ms=5)
        a.set_title(title)
        a.set_xlabel("x (mm)")
        a.set_ylabel("y (mm)")
        fig.colorbar(im, ax=a, fraction=0.046, label="c (m/s)")

    ax[3].semilogy(np.array(history) / history[0], "o-")
    ax[3].set_title("Misfit convergence")
    ax[3].set_xlabel("iteration")
    ax[3].set_ylabel("normalised misfit")
    ax[3].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "reconstruction.png")
    fig.savefig(out, dpi=130)
    print(f"saved {os.path.normpath(out)}")
    print(f"misfit reduced to {history[-1] / history[0] * 100:.1f}% of initial")


if __name__ == "__main__":
    main()
