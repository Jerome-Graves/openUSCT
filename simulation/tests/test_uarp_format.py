"""Round-trip test for the UARP / UDSP v4.0 acquisition format.

Simulates a ring-matrix full-matrix-capture acquisition, writes it in UDSP v4.0
format, reads it back, and checks the channel data, geometry, timing, and
transmit mapping survive. Also checks the file has the expected UDSP group
layout, so it is structurally the format the Leeds UARP tools expect.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ringfwi import phantom
from ringfwi.acquire import simulate_dataset
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker
from ringfwi.uarp_format import from_uarp_set, to_uarp_set


def test_uarp_roundtrip_and_layout():
    h = 2.0e-3
    cyl = CylinderArray(n_rings=2, per_ring=8, radius_m=0.014, height_m=0.010,
                        domain_m=0.032, h=h)
    n = cyl.n
    dt = 2.0e-7
    nt = 120
    wavelet = ricker(nt, dt, 0.3e6)

    c = phantom.cylinder_background((n, n, n), 3000.0, 1480.0, 0.011, h)
    c = phantom.add_sphere(c, (0.5, 0.5, 0.5), 0.004, 2700.0, h)
    ds = simulate_dataset(c, cyl, wavelet, dt, nominal_speed_m_s=1480.0,
                          src_list=[0, 4, 8, 12])

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "acq_udsp.h5")
        to_uarp_set(ds, path)

        # Structural check: the UDSP v4.0 groups exist.
        import h5py
        with h5py.File(path, "r") as f:
            assert f.attrs["Version"] == "4.0"
            s0 = f["Series/Series000"]
            assert "Dimensions/Value" in s0
            assert float(s0["Dimensions/Dimension1"].attrs["Uniform.Stride"]) == dt
            assert "Dimensions/Dimension2/Sparse.Points" in s0
            assert int(s0["Frames"].attrs["Size"]) == 4
            assert "Frames/Frame000000" in s0

        loaded = from_uarp_set(path)

    assert loaded.n_tx == 4
    assert loaded.n_samples == nt
    assert loaded.geometry.dim == 3
    assert np.allclose(loaded.data, ds.data, rtol=1e-5, atol=1e-8)
    assert np.allclose(loaded.geometry.element_pos, ds.geometry.element_pos)
    assert abs(loaded.dt - dt) < 1e-15
    assert list(loaded.tx_elements) == [0, 4, 8, 12]
    print(f"UARP v4.0 round-trip OK: {loaded.n_tx} frames, {loaded.geometry.n_elements} elements, "
          f"dt {loaded.dt*1e9:.1f} ns")


if __name__ == "__main__":
    test_uarp_roundtrip_and_layout()
    print("UARP format round-trip passed")
