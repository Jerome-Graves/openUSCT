"""Full pipeline demonstration: acquire -> Dataset -> save -> load -> reconstruct.

Shows the whole platform loop through the portable HDF5 format:
  1. simulate a full-matrix-capture acquisition (the "acquire" backend),
  2. save it to a portable HDF5 dataset and read it back,
  3. reconstruct it with FWI through the plugin interface.

The reconstruction call is identical to what a hardware-acquired dataset would
use, which is the point of the shared format.

Run:  python examples/run_pipeline_demo.py
Output: figures/pipeline.png  (and acquisition.h5)
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import phantom, plugins
from ringfwi.acquire import simulate_dataset
from ringfwi.dataset import Dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def main():
    here = os.path.dirname(__file__)
    h = 1.0e-3
    domain = 0.050
    ring = RingArray(n_elements=16, radius_m=0.020, domain_m=domain, h=h)
    n = ring.n
    dt = 8.0e-8
    nt = 500
    wavelet = ricker(nt, dt, 0.3e6)

    spec_radius = 0.016
    flaw = (0.60, 0.50)
    c_bg = phantom.coupling_background((n, n), 3000.0, 1480.0, spec_radius, h)
    c_true = phantom.add_inclusion(c_bg, flaw, 0.005, 2600.0, h)

    # 1. Acquire.
    print("1. simulating acquisition ...")
    ds = simulate_dataset(c_true, ring, wavelet, dt, nominal_speed_m_s=3000.0,
                          description="pipeline demo specimen with flaw")

    # 2. Save to portable HDF5 and read back.
    path = os.path.join(here, "..", "acquisition.h5")
    ds.save(path)
    loaded = Dataset.load(path)
    print(f"2. saved and reloaded {os.path.normpath(path)} "
          f"(data {loaded.data.shape}, {loaded.geometry.dim}D {loaded.geometry.array_type})")

    # 3. Reconstruct via the FWI plugin, starting from the flaw-free background.
    print("3. reconstructing with the 'fwi' plugin ...")
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2500.0))
    result = plugins.run("fwi", loaded, start_c=c_bg, m_bounds=m_bounds,
                         n_iter=12, step_frac=0.03, verbose=True)

    history = result["history"]
    lo, hi = result["extent"]
    ext = [lo * 1e3, hi * 1e3, lo * 1e3, hi * 1e3]
    print(f"   misfit reduced to {history[-1]/history[0]*100:.1f}% of initial")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
    vmin, vmax = 2500, 3100
    for a, field, title in ((ax[0], c_true, "True model"),
                            (ax[1], c_bg, "Starting model"),
                            (ax[2], result["c"], "FWI via pipeline")):
        im = a.imshow(field, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
        a.set_title(title); a.set_xlabel("mm"); a.set_ylabel("mm")
        fig.colorbar(im, ax=a, fraction=0.046, label="c (m/s)")

    ax[3].semilogy(np.array(history) / history[0], "o-")
    ax[3].set_title("Misfit convergence")
    ax[3].set_xlabel("iteration"); ax[3].set_ylabel("normalised misfit")
    ax[3].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(here, "..", "figures", "pipeline.png")
    fig.savefig(out, dpi=125)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
