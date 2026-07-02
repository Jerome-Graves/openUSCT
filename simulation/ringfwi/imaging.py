"""Delay-and-sum imaging: the total focusing method (TFM).

TFM is the reference full-matrix-capture imaging algorithm. For each image point
it sums the contribution of every transmit/receive pair, sampling each trace at
the two-way travel time from the transmitting element to the point to the
receiving element, assuming a constant background speed:

    I(r) = sum_{tx, rx}  s_{tx,rx}( (|r - p_tx| + |r - p_rx|) / c )

Using the analytic (Hilbert) signal and taking the magnitude gives an envelope
image. This is qualitative reflectivity imaging, complementary to the
quantitative sound-speed maps from FWI, and it is fast. It is dimension-general:
element positions and image points may be 2D or 3D.
"""

from __future__ import annotations

import numpy as np


def image_axes(dataset, npix, half_size=None):
    """Return the per-axis coordinate vectors of a centred image grid (metres)."""
    half = half_size if half_size is not None else dataset.geometry.radius_m * 0.95
    return [np.linspace(-half, half, npix) for _ in range(dataset.geometry.dim)]


def tfm(dataset, npix=120, half_size=None, envelope=True):
    """Total focusing method image of a :class:`~ringfwi.dataset.Dataset`.

    Parameters
    ----------
    npix : int
        Pixels per axis of the (square/cubic) image.
    half_size : float, optional
        Half-width of the image region in metres (defaults to 0.95 * ring radius).
    envelope : bool
        If True, return the analytic-signal envelope magnitude.

    Returns
    -------
    image : ndarray
        2D or 3D image, shape ``(npix,) * dim``.
    axes : list of ndarray
        The coordinate vector along each axis (metres).
    """
    from scipy.signal import hilbert

    dim = dataset.geometry.dim
    axes = image_axes(dataset, npix, half_size)
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)  # (P, dim)

    epos = dataset.geometry.element_pos                 # (N, dim)
    tx_elements = np.asarray(dataset.tx_elements)
    c = dataset.nominal_speed_m_s
    fs = dataset.sample_rate_hz
    nt = dataset.n_samples

    # Distance from every image point to every element: (P, N).
    d = np.sqrt(((pts[:, None, :] - epos[None, :, :]) ** 2).sum(axis=2))

    analytic = hilbert(dataset.data, axis=1)            # (n_tx, nt, n_rx) complex
    acc = np.zeros(pts.shape[0], dtype=complex)

    for i, te in enumerate(tx_elements):
        d_tx = d[:, te]
        for j in range(dataset.n_rx):
            t = (d_tx + d[:, j]) / c
            k = np.round(t * fs).astype(np.int64)
            valid = (k >= 0) & (k < nt)
            acc[valid] += analytic[i, k[valid], j]

    image = np.abs(acc) if envelope else acc.real
    return image.reshape([npix] * dim), axes
