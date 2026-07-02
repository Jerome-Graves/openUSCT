"""Simulation backend of the acquisition API.

This is the software equivalent of pressing "acquire" on the hardware: it runs
the wave solver and returns a :class:`~ringfwi.dataset.Dataset` in exactly the
same form the physical board will produce. Application and imaging code targets
the Dataset, so nothing downstream needs to know whether the data came from a
solver or a PCB. That is the hardware abstraction layer, seen from the software
side.
"""

from __future__ import annotations

from .dataset import ArrayGeometry, Dataset
from .fwi import forward_fmc
from .phantom import velocity_to_m


def simulate_dataset(
    c_model,
    array_geom,
    wavelet,
    dt,
    nominal_speed_m_s,
    src_list=None,
    store_ground_truth=True,
    description="",
):
    """Simulate a full-matrix-capture acquisition and return a Dataset.

    Works in 2D or 3D depending on the array passed.

    Parameters
    ----------
    c_model : ndarray
        True sound-speed field (m/s); 2D for a ring, 3D for a cylinder.
    array_geom : RingArray or CylinderArray
        Grid-mapped array used by the solver.
    wavelet : (nt,) ndarray
        Transmit wavelet.
    dt : float
        Time step (seconds); the dataset sample rate is 1 / dt.
    nominal_speed_m_s : float
        Assumed background speed recorded for delay-based imaging.
    """
    m = velocity_to_m(c_model)
    data = forward_fmc(m, array_geom, wavelet, dt, array_geom.h, len(wavelet), src_list=src_list)

    geom = ArrayGeometry(
        element_pos=array_geom.element_positions,
        radius_m=array_geom.radius_m,
        centre_freq_hz=_peak_freq(wavelet, dt),
        array_type=getattr(array_geom, "array_type", "ring"),
    )

    tx_elements = (list(range(array_geom.n_elements)) if src_list is None
                   else list(src_list))
    gt = {"c": c_model, "h_m": array_geom.h} if store_ground_truth else None
    return Dataset(
        geometry=geom,
        data=data,
        sample_rate_hz=1.0 / dt,
        tx_wavelet=wavelet,
        tx_centre_freq_hz=_peak_freq(wavelet, dt),
        nominal_speed_m_s=nominal_speed_m_s,
        tx_elements=tx_elements,
        ground_truth=gt,
        metadata={"description": description, "created_by": "OpenUSCT simulator"},
    )


def _peak_freq(wavelet, dt):
    """Estimate the wavelet centre frequency from its amplitude spectrum."""
    import numpy as np

    spec = np.abs(np.fft.rfft(wavelet))
    freqs = np.fft.rfftfreq(len(wavelet), dt)
    return float(freqs[int(np.argmax(spec))])
