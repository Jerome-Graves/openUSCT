"""Round-trip test for the portable HDF5 Dataset format.

Simulates a small acquisition, writes it to HDF5, reads it back, and checks the
geometry, channel data, and ground truth survive unchanged. This is what lets
Python, MATLAB, and C++ exchange the same data.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ringfwi import phantom
from ringfwi.acquire import simulate_dataset
from ringfwi.dataset import Dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker


def test_dataset_hdf5_roundtrip():
    h = 1.0e-3
    ring = RingArray(n_elements=8, radius_m=0.024, domain_m=0.06, h=h)
    n = ring.n
    dt = 1.2e-7
    nt = 300
    wavelet = ricker(nt, dt, 0.4e6)

    c = phantom.coupling_background((n, n), 3000.0, 1480.0, 0.02, h)
    c = phantom.add_inclusion(c, (0.5, 0.5), 0.006, 2700.0, h)

    ds = simulate_dataset(c, ring, wavelet, dt, nominal_speed_m_s=1480.0,
                          src_list=[0, 2, 4], description="roundtrip test")

    assert ds.data.shape == (3, nt, 8)
    assert ds.geometry.n_elements == 8

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ds.h5")
        ds.save(path)
        loaded = Dataset.load(path)

    # float32 on disk, so compare with a tolerance
    assert np.allclose(loaded.data, ds.data, rtol=1e-5, atol=1e-8)
    assert np.allclose(loaded.geometry.element_pos, ds.geometry.element_pos)
    assert loaded.geometry.dim == 2
    assert loaded.sample_rate_hz == ds.sample_rate_hz
    assert loaded.nominal_speed_m_s == 1480.0
    assert loaded.ground_truth is not None
    assert np.allclose(loaded.ground_truth["c"], c)
    print(f"2D roundtrip OK: data {loaded.data.shape}, centre_freq "
          f"{loaded.tx_centre_freq_hz/1e6:.3f} MHz")


def test_dataset_hdf5_roundtrip_3d():
    from ringfwi.geometry import CylinderArray

    h = 2.0e-3
    cyl = CylinderArray(n_rings=2, per_ring=6, radius_m=0.012,
                        height_m=0.010, domain_m=0.03, h=h)
    n = cyl.n
    dt = 2.0e-7
    nt = 120
    wavelet = ricker(nt, dt, 0.3e6)

    c = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.009, h)
    c = phantom.add_sphere(c, (0.5, 0.5, 0.5), 0.004, 2700.0, h)

    ds = simulate_dataset(c, cyl, wavelet, dt, nominal_speed_m_s=1480.0, src_list=[0, 6])

    assert ds.geometry.dim == 3
    assert ds.geometry.element_pos.shape == (12, 3)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ds3d.h5")
        ds.save(path)
        loaded = Dataset.load(path)

    assert loaded.geometry.dim == 3
    assert loaded.geometry.array_type == "cylinder"
    assert np.allclose(loaded.data, ds.data, rtol=1e-5, atol=1e-8)
    assert np.allclose(loaded.ground_truth["c"], c)
    assert loaded.ground_truth["c"].ndim == 3
    print(f"3D roundtrip OK: data {loaded.data.shape}, geometry {loaded.geometry.element_pos.shape}")


if __name__ == "__main__":
    test_dataset_hdf5_roundtrip()
    test_dataset_hdf5_roundtrip_3d()
    print("dataset roundtrips passed")
