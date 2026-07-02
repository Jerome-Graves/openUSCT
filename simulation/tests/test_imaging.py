"""Imaging and plugin tests.

- TFM must focus a point scatterer at its true location: this validates the
  travel-time geometry and delay indexing.
- The plugin registry must register, list, and run algorithms on a Dataset.
"""

from __future__ import annotations

import numpy as np

from ringfwi import imaging, phantom, plugins
from ringfwi.acquire import simulate_dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def _point_scatterer_dataset():
    h = 1.0e-3
    ring = RingArray(n_elements=16, radius_m=0.020, domain_m=0.050, h=h)
    n = ring.n
    dt = 1.5e-7
    nt = 320
    wavelet = ricker(nt, dt, 0.5e6)

    # Homogeneous water with one strong, small scatterer off-centre, plus a
    # matching reference with no scatterer. Subtracting isolates the scattered
    # field so TFM focuses the scatterer rather than the direct wave.
    c_ref = np.full((n, n), 1480.0)
    c_scat = phantom.add_inclusion(c_ref, (0.62, 0.50), 0.003, 3200.0, h)

    ds = simulate_dataset(c_scat, ring, wavelet, dt, nominal_speed_m_s=1480.0)
    ds_ref = simulate_dataset(c_ref, ring, wavelet, dt, nominal_speed_m_s=1480.0)
    ds.data = ds.data - ds_ref.data  # scattered field only

    domain = (n - 1) * h
    true_xy = np.array([0.62 * domain - domain / 2, 0.50 * domain - domain / 2])
    return ds, true_xy


def test_tfm_focuses_point_scatterer():
    ds, true_xy = _point_scatterer_dataset()
    image, axes = imaging.tfm(ds, npix=81, half_size=0.018)

    ix, iy = np.unravel_index(int(np.argmax(image)), image.shape)
    peak_xy = np.array([axes[0][ix], axes[1][iy]])

    err = float(np.hypot(*(peak_xy - true_xy)))
    print(f"true {true_xy*1e3} mm, peak {peak_xy*1e3} mm, err {err*1e3:.2f} mm")
    assert err < 0.004  # within 4 mm


def test_plugin_registry():
    assert "tfm" in plugins.available()

    @plugins.register("dummy_energy", description="total channel energy")
    def _dummy(dataset, **params):
        return float(np.sum(dataset.data ** 2))

    ds, _ = _point_scatterer_dataset()
    val = plugins.run("dummy_energy", ds)
    assert val > 0.0

    img = plugins.run("tfm", ds, npix=41, half_size=0.018)
    assert img.shape == (41, 41)


if __name__ == "__main__":
    test_tfm_focuses_point_scatterer()
    test_plugin_registry()
    print("imaging + plugin tests passed")
