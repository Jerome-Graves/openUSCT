"""Worked example: bring your own algorithm.

Registers a custom imaging algorithm (a bandpass-filtered TFM) through the
plugin interface and runs it on the same Dataset as the built-in TFM. This is
the intended extension point: a researcher drops in their own DSP or imaging
method and it runs on equal footing with the built-ins, on identical data.

Run:  python examples/custom_plugin.py
"""

from __future__ import annotations

import copy

import numpy as np

from ringfwi import imaging, phantom, plugins
from ringfwi.acquire import simulate_dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


@plugins.register("bandpass_tfm", description="bandpass filter the channels, then TFM")
def bandpass_tfm(dataset, band=(0.15e6, 0.45e6), npix=120, half_size=None):
    """Custom algorithm: pulse-shaping bandpass before delay-and-sum."""
    from scipy.signal import butter, filtfilt

    fs = dataset.sample_rate_hz
    b, a = butter(4, [2 * band[0] / fs, 2 * band[1] / fs], btype="band")
    filtered = filtfilt(b, a, dataset.data, axis=1)

    ds2 = copy.copy(dataset)          # same geometry and parameters ...
    ds2.data = filtered               # ... but the filtered channel data
    image, _axes = imaging.tfm(ds2, npix=npix, half_size=half_size)
    return image


def main():
    h = 1.0e-3
    ring = RingArray(n_elements=24, radius_m=0.024, domain_m=0.060, h=h)
    n = ring.n
    dt = 7.0e-8
    nt = 700
    wavelet = ricker(nt, dt, 0.30e6)

    c = np.full((n, n), 2500.0)
    for f in [(0.60, 0.52), (0.40, 0.58)]:
        c = phantom.add_inclusion(c, f, 0.0025, 3500.0, h)
    c_ref = np.full((n, n), 2500.0)

    ds = simulate_dataset(c, ring, wavelet, dt, nominal_speed_m_s=2500.0)
    ds_ref = simulate_dataset(c_ref, ring, wavelet, dt, nominal_speed_m_s=2500.0)
    ds.data = ds.data - ds_ref.data

    print("registered algorithms:")
    for name, desc in plugins.available().items():
        print(f"  {name:14s} {desc}")

    built_in = plugins.run("tfm", ds, npix=100, half_size=0.014)
    custom = plugins.run("bandpass_tfm", ds, npix=100, half_size=0.014)

    print(f"\nbuilt-in tfm       -> image {built_in.shape}, peak {built_in.max():.3e}")
    print(f"custom bandpass_tfm -> image {custom.shape}, peak {custom.max():.3e}")
    print("\nboth ran on the same Dataset through the plugin interface.")


if __name__ == "__main__":
    main()
