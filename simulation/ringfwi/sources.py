"""Source wavelets."""

from __future__ import annotations

import numpy as np


def ricker(nt, dt, f0, t0=None):
    """Ricker (Mexican-hat) wavelet, the standard FWI test source.

    Parameters
    ----------
    nt : int
        Number of samples.
    dt : float
        Time step, seconds.
    f0 : float
        Centre frequency, Hz.
    t0 : float, optional
        Time shift of the peak; defaults to 1 / f0 so the wavelet is causal.
    """
    if t0 is None:
        t0 = 1.0 / f0
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gabor(nt, dt, f0, frac_bw=0.6, t0=None):
    """Gaussian-modulated cosine (Gabor) wavelet with a controllable bandwidth.

    ``frac_bw`` is the fractional -6 dB bandwidth: smaller means a longer, more
    narrow-band pulse; larger means a shorter, broadband pulse. This models a
    transducer of a given bandwidth more realistically than the fixed-shape
    Ricker.
    """
    sigma_f = frac_bw * f0 / 2.3548           # FWHM -> Gaussian sigma (frequency)
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    if t0 is None:
        t0 = 3.0 * sigma_t
    t = np.arange(nt) * dt - t0
    return np.exp(-t ** 2 / (2.0 * sigma_t ** 2)) * np.cos(2.0 * np.pi * f0 * t)


def hann_tone_burst(nt, dt, f0, n_cycles=1, t0=None):
    """Hann-tapered sinusoidal burst.

    Matches the single-cycle Hann-tapered excitation used in the IUS 2026
    ice-core simulation framework, so forward-model behaviour is comparable.
    """
    if t0 is None:
        t0 = 0.0
    t = np.arange(nt) * dt
    dur = n_cycles / f0
    w = np.zeros(nt)
    mask = (t >= t0) & (t <= t0 + dur)
    tau = t[mask] - t0
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau / dur))
    w[mask] = env * np.sin(2.0 * np.pi * f0 * tau)
    return w
