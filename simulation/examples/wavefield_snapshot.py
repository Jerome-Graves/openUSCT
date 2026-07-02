"""Visualise wave propagation through the ring-array specimen.

Fires one element and shows the pressure field at three instants as the
wavefront crosses the specimen and scatters off the inclusion.

Run:  python examples/wavefield_snapshot.py
Output: figures/wavefield.png
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import fwi, phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def main():
    h = 6.0e-4
    domain = 0.06
    geom = RingArray(n_elements=16, radius_m=0.024, domain_m=domain, h=h)
    n = geom.n

    dt = 7.0e-8
    nt = 800
    wavelet = ricker(nt, dt, 0.25e6)

    c = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.020, h)
    c = phantom.add_inclusion(c, (0.58, 0.46), 0.007, 2650.0, h)
    m = phantom.velocity_to_m(c)

    hist = fwi.simulate_wavefield(m, geom, 0, wavelet, dt, h, nt)

    frames = [180, 360, 560]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ext = [0, domain * 1e3, 0, domain * 1e3]
    for a, fr in zip(ax, frames):
        amp = np.max(np.abs(hist[fr])) + 1e-30
        a.imshow(hist[fr], origin="lower", extent=ext, cmap="seismic", vmin=-amp, vmax=amp)
        a.plot(geom.xy[:, 0] * 1e3, geom.xy[:, 1] * 1e3, "k.", ms=4)
        a.set_title(f"t = {fr * dt * 1e6:.1f} us")
        a.set_xlabel("x (mm)")
        a.set_ylabel("y (mm)")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "wavefield.png")
    fig.savefig(out, dpi=130)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
