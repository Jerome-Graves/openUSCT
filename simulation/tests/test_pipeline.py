"""Closed-loop pipeline test.

Simulate an acquisition, write it to HDF5, read it back, and reconstruct it with
FWI through the plugin interface, all via the portable Dataset. This proves the
whole chain (acquire -> Dataset -> save -> load -> process) holds together and
that FWI runs on the same portable object TFM does.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ringfwi import phantom, plugins
from ringfwi.acquire import simulate_dataset
from ringfwi.dataset import Dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def test_pipeline_simulate_save_load_fwi():
    h = 1.25e-3
    ring = RingArray(n_elements=12, radius_m=0.020, domain_m=0.050, h=h)
    n = ring.n
    dt = 1.2e-7
    nt = 360
    wavelet = ricker(nt, dt, 0.3e6)

    spec_radius = 0.016
    flaw = (0.60, 0.50)
    c_bg = phantom.coupling_background((n, n), 3000.0, 1480.0, spec_radius, h)
    c_true = phantom.add_inclusion(c_bg, flaw, 0.005, 2600.0, h)

    ds = simulate_dataset(c_true, ring, wavelet, dt, nominal_speed_m_s=3000.0)

    # Round-trip through the portable format.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "acq.h5")
        ds.save(path)
        loaded = Dataset.load(path)

    # Reconstruct via the plugin, starting from the flaw-free background.
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2500.0))
    result = plugins.run("fwi", loaded, start_c=c_bg, m_bounds=m_bounds,
                         n_iter=6, step_frac=0.04)

    hist = result["history"]
    assert hist[-1] < 0.5 * hist[0], "FWI did not reduce the misfit"

    # The flaw voxel should be recovered as lower velocity than the background.
    ix = int(round(flaw[0] * (n - 1)))
    iy = int(round(flaw[1] * (n - 1)))
    c_rec = result["c"]
    print(f"background 3000, recovered at flaw {c_rec[iy, ix]:.0f} m/s, "
          f"misfit {hist[-1]/hist[0]*100:.1f}%")
    assert c_rec[iy, ix] < 2900.0


if __name__ == "__main__":
    test_pipeline_simulate_save_load_fwi()
    print("pipeline test passed")
