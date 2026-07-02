"""Portable acquisition dataset: the shared contract across the platform.

A :class:`Dataset` is the raw product of an acquisition, whether it came from
the simulator or (later) from the hardware. It bundles the array geometry, the
full-matrix-capture channel data, and the acquisition parameters, and it
serialises to a single self-describing HDF5 file so that Python, MATLAB, and C++
all read and write the same thing.

The geometry is stored in physical coordinates (metres, origin at the array
centre) and is deliberately independent of any simulation grid, so the same
dataset images identically no matter how it was produced. The HDF5 layout is
specified in software/schema/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = "1.0"


@dataclass
class ArrayGeometry:
    """Transducer array geometry in physical coordinates.

    element_pos : (n_elements, dim) ndarray
        Element positions in metres, origin at the array centre. ``dim`` is 2
        for a planar ring or 3 for a cylindrical array.
    """

    element_pos: np.ndarray
    radius_m: float
    centre_freq_hz: float
    array_type: str = "ring"

    @property
    def n_elements(self):
        return self.element_pos.shape[0]

    @property
    def dim(self):
        return self.element_pos.shape[1]

    @classmethod
    def ring(cls, n_elements, radius_m, centre_freq_hz):
        """Evenly spaced planar ring of elements centred on the origin (2D)."""
        ang = np.linspace(0.0, 2.0 * np.pi, n_elements, endpoint=False)
        xy = np.column_stack([radius_m * np.cos(ang), radius_m * np.sin(ang)])
        return cls(xy, radius_m, centre_freq_hz, "ring")

    @classmethod
    def cylinder(cls, n_rings, per_ring, radius_m, height_m, centre_freq_hz):
        """Rings of elements stacked along the z axis, centred on the origin (3D)."""
        zs = np.array([0.0]) if n_rings == 1 else np.linspace(-height_m / 2, height_m / 2, n_rings)
        ang = np.linspace(0.0, 2.0 * np.pi, per_ring, endpoint=False)
        pos = [[radius_m * np.cos(a), radius_m * np.sin(a), z] for z in zs for a in ang]
        return cls(np.asarray(pos), radius_m, centre_freq_hz, "cylinder")


@dataclass
class Dataset:
    """A full-matrix-capture acquisition plus its geometry and parameters.

    data : (n_tx, n_samples, n_rx) ndarray
        Channel time series; element ``i`` transmits, element ``j`` records.
    sample_rate_hz : float
    tx_wavelet : (n_samples,) ndarray
    tx_centre_freq_hz : float
    nominal_speed_m_s : float
        Assumed background sound speed used by delay-based imaging.
    ground_truth : dict or None
        Optional simulation ground truth: {"c": (ny, nx), "h_m": float}.
    """

    geometry: ArrayGeometry
    data: np.ndarray
    sample_rate_hz: float
    tx_wavelet: np.ndarray
    tx_centre_freq_hz: float
    nominal_speed_m_s: float
    tx_elements: np.ndarray | None = None
    ground_truth: dict | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # tx_elements maps each transmit index to an element index in the
        # geometry. Defaults to "every element transmitted, in order".
        if self.tx_elements is None:
            self.tx_elements = np.arange(self.data.shape[0])
        else:
            self.tx_elements = np.asarray(self.tx_elements)

    @property
    def n_tx(self):
        return self.data.shape[0]

    @property
    def n_samples(self):
        return self.data.shape[1]

    @property
    def n_rx(self):
        return self.data.shape[2]

    @property
    def dt(self):
        return 1.0 / self.sample_rate_hz

    # -- serialisation ---------------------------------------------------------
    def save(self, path):
        """Write the dataset to a self-describing HDF5 file."""
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["description"] = self.metadata.get("description", "")
            f.attrs["created_by"] = self.metadata.get("created_by", "OpenUSCT")

            g = f.create_group("geometry")
            g.attrs["n_elements"] = self.geometry.n_elements
            g.attrs["radius_m"] = self.geometry.radius_m
            g.attrs["centre_freq_hz"] = self.geometry.centre_freq_hz
            g.attrs["array_type"] = self.geometry.array_type
            g.create_dataset("element_pos", data=self.geometry.element_pos)

            a = f.create_group("acquisition")
            a.attrs["sample_rate_hz"] = self.sample_rate_hz
            a.attrs["tx_centre_freq_hz"] = self.tx_centre_freq_hz
            a.attrs["nominal_speed_m_s"] = self.nominal_speed_m_s
            a.create_dataset("data", data=self.data.astype(np.float32), compression="gzip")
            a.create_dataset("tx_wavelet", data=self.tx_wavelet)
            a.create_dataset("tx_elements", data=np.asarray(self.tx_elements))

            if self.ground_truth is not None:
                t = f.create_group("ground_truth")
                t.attrs["h_m"] = self.ground_truth["h_m"]
                t.create_dataset("c", data=self.ground_truth["c"], compression="gzip")

    @classmethod
    def load(cls, path):
        """Read a dataset from an HDF5 file written by :meth:`save`."""
        import h5py

        with h5py.File(path, "r") as f:
            g = f["geometry"]
            geom = ArrayGeometry(
                element_pos=g["element_pos"][()],
                radius_m=float(g.attrs["radius_m"]),
                centre_freq_hz=float(g.attrs["centre_freq_hz"]),
                array_type=str(g.attrs["array_type"]),
            )
            a = f["acquisition"]
            gt = None
            if "ground_truth" in f:
                t = f["ground_truth"]
                gt = {"c": t["c"][()], "h_m": float(t.attrs["h_m"])}
            tx_elements = a["tx_elements"][()] if "tx_elements" in a else None
            return cls(
                geometry=geom,
                data=a["data"][()],
                sample_rate_hz=float(a.attrs["sample_rate_hz"]),
                tx_wavelet=a["tx_wavelet"][()],
                tx_centre_freq_hz=float(a.attrs["tx_centre_freq_hz"]),
                nominal_speed_m_s=float(a.attrs["nominal_speed_m_s"]),
                tx_elements=tx_elements,
                ground_truth=gt,
                metadata={
                    "description": str(f.attrs.get("description", "")),
                    "created_by": str(f.attrs.get("created_by", "")),
                    "format_version": str(f.attrs.get("format_version", "")),
                },
            )
