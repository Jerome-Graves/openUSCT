"""UARP / UDSP v4.0 HDF5 format I/O.

Reads and writes acquisitions in the UARP UDSP data format (schema v4.0), so a
ring-matrix full-matrix-capture acquisition produced by OpenUSCT is stored in the
format the Leeds UARP toolbox ecosystem uses. This is a clean-room
implementation that conforms to the published schema layout (Set -> Series ->
Dimensions + Frames); it does not use any UARP source code.

Layout (schema v4.0):

    /                                Version, Timestamp, Description
    /Series/Series000                Description, State, Platform, Source
      /Dimensions
        /Value                       Description, Unit, Scale (the sample value)
        /Dimension1 (fast time)      Uniform.Size, Uniform.Stride, Uniform.Offset
          /DimensionQuantity         Description, Unit, Scale
        /Dimension2 (rx element)     Sparse.Points  (element positions)
          /DimensionQuantity         Description, Unit, Scale
      /Frames                        Size
        /Frame000000 ...             one 2D [nt, n_rx] array per transmit; Delay

Because a real acquisition contains only the captured data, timing, and
geometry, OpenUSCT extras (the transmit wavelet, the assumed imaging speed, and
simulation ground truth) are not part of the UARP file; they are supplied
separately when processing.
"""

from __future__ import annotations

import numpy as np

from .dataset import ArrayGeometry, Dataset

SCHEMA_VERSION = "4.0"


def to_uarp_set(dataset, path, description="OpenUSCT full-matrix-capture acquisition",
                timestamp="1970-01-01T00:00:00"):
    """Write a :class:`~ringfwi.dataset.Dataset` as a UDSP v4.0 HDF5 Set."""
    import h5py

    ds = dataset
    tx = np.asarray(ds.tx_elements)
    with h5py.File(path, "w") as f:
        f.attrs["Version"] = SCHEMA_VERSION
        f.attrs["Timestamp"] = timestamp
        f.attrs["Description"] = description

        s0 = f.create_group("Series").create_group("Series000")
        s0.attrs["Description"] = "Full-matrix capture channel data"
        s0.attrs["State"] = "Analytical" if ds.ground_truth is not None else "Measured"
        s0.attrs["Platform"] = "OpenUSCT"
        s0.attrs["Source"] = "Simulator"

        dims = s0.create_group("Dimensions")
        val = dims.create_group("Value")
        val.attrs["Description"] = "Signal amplitude"
        val.attrs["Unit"] = "Amplitude"
        val.attrs["Scale"] = "None"

        d1 = dims.create_group("Dimension1")
        d1.attrs["Description"] = "Fast time"
        d1.attrs["Uniform.Size"] = ds.n_samples
        d1.attrs["Uniform.Stride"] = ds.dt
        d1.attrs["Uniform.Offset"] = 0.0
        q1 = d1.create_group("DimensionQuantity")
        q1.attrs["Description"] = "Time"; q1.attrs["Unit"] = "Seconds"; q1.attrs["Scale"] = "None"

        d2 = dims.create_group("Dimension2")
        d2.attrs["Description"] = "Receive element"
        d2.create_dataset("Sparse.Points", data=ds.geometry.element_pos)
        q2 = d2.create_group("DimensionQuantity")
        q2.attrs["Description"] = "Element position"; q2.attrs["Unit"] = "Metres"; q2.attrs["Scale"] = "None"

        frames = s0.create_group("Frames")
        frames.attrs["Size"] = ds.n_tx
        for fidx in range(ds.n_tx):
            fr = frames.create_dataset("Frame%06d" % fidx,
                                       data=ds.data[fidx].astype(np.float32), compression="gzip")
            fr.attrs["Description"] = "tx_element=%d" % int(tx[fidx])
            fr.attrs["Delay"] = 0.0


def from_uarp_set(path, nominal_speed_m_s=1500.0):
    """Read a UDSP v4.0 HDF5 Set into a :class:`~ringfwi.dataset.Dataset`."""
    import h5py

    with h5py.File(path, "r") as f:
        s0 = f["Series"]["Series000"]
        dims = s0["Dimensions"]
        nt = int(dims["Dimension1"].attrs["Uniform.Size"])
        dt = float(dims["Dimension1"].attrs["Uniform.Stride"])
        epos = dims["Dimension2"]["Sparse.Points"][()]

        frames = s0["Frames"]
        keys = sorted(k for k in frames.keys() if k.startswith("Frame"))
        data = np.stack([frames[k][()] for k in keys], axis=0)

        tx_elements = []
        for k in keys:
            desc = frames[k].attrs.get("Description", "")
            tx_elements.append(int(str(desc).split("=")[1]) if "=" in str(desc) else len(tx_elements))

    radius = float(np.max(np.linalg.norm(epos, axis=1)))
    geom = ArrayGeometry(element_pos=epos, radius_m=radius, centre_freq_hz=0.0,
                         array_type="cylinder" if epos.shape[1] == 3 else "ring")
    return Dataset(
        geometry=geom,
        data=data,
        sample_rate_hz=1.0 / dt,
        tx_wavelet=np.zeros(nt),
        tx_centre_freq_hz=0.0,
        nominal_speed_m_s=nominal_speed_m_s,
        tx_elements=np.asarray(tx_elements),
        ground_truth=None,
    )
