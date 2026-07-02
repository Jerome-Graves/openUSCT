"""Total focusing method (TFM) imaging demonstration.

TFM is fast qualitative reflectivity imaging: for each pixel it delay-and-sums
every transmit/receive pair assuming a constant background speed. It is the
natural complement to FWI, which is slower but reconstructs the actual
sound-speed map.

This demo images small scatterers ("flaws") in a fairly uniform medium, the
regime where TFM's constant-speed assumption holds. It uses reference
subtraction to isolate the scattered field. Note the limitation this illustrates:
in a strongly heterogeneous medium (for example a specimen in a water bath) a
single assumed speed mis-focuses, which is exactly the case FWI is built for.

Run:  python examples/run_imaging_demo.py
Output: figures/tfm.png
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import imaging, phantom
from ringfwi.acquire import simulate_dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def main():
    h = 6.0e-4
    domain = 0.06
    ring = RingArray(n_elements=24, radius_m=0.024, domain_m=domain, h=h)
    n = ring.n
    dt = 7.0e-8
    nt = 800
    wavelet = ricker(nt, dt, 0.35e6)

    c_bg = 2500.0
    flaws = [(0.60, 0.52), (0.40, 0.60), (0.52, 0.36)]

    c_ref = np.full((n, n), c_bg)
    c_flaw = c_ref
    for f in flaws:
        c_flaw = phantom.add_inclusion(c_flaw, f, 0.0025, 3500.0, h)

    ds = simulate_dataset(c_flaw, ring, wavelet, dt, nominal_speed_m_s=c_bg)
    ds_ref = simulate_dataset(c_ref, ring, wavelet, dt, nominal_speed_m_s=c_bg)
    ds.data = ds.data - ds_ref.data  # scattered field only

    img, axes = imaging.tfm(ds, npix=160, half_size=0.014)

    half = (n - 1) * h / 2
    true_xy = [(fx * (n - 1) * h - half, fy * (n - 1) * h - half) for fx, fy in flaws]

    ext = [axes[0][0] * 1e3, axes[0][-1] * 1e3, axes[1][0] * 1e3, axes[1][-1] * 1e3]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    ax[0].imshow(c_flaw, origin="lower", extent=[0, domain * 1e3, 0, domain * 1e3], cmap="viridis")
    ax[0].plot(ring.xy[:, 0] * 1e3, ring.xy[:, 1] * 1e3, "w.", ms=5)
    ax[0].set_title("True model: 3 scatterers")

    ax[1].imshow(img.T, origin="lower", extent=ext, cmap="inferno")
    for (tx, ty) in true_xy:
        ax[1].plot(tx * 1e3, ty * 1e3, "co", mfc="none", ms=16, mew=2)
    ax[1].set_title("TFM image of scattered field")

    for a in ax:
        a.set_xlabel("mm"); a.set_ylabel("mm")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "tfm.png")
    fig.savefig(out, dpi=130)
    print(f"saved {os.path.normpath(out)}  (circles = true scatterer locations)")


if __name__ == "__main__":
    main()
