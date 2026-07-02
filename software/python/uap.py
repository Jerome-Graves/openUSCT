"""uap: Python front end for the libuap C++ core.

Provides the same forward_fmc / misfit_and_gradient signatures as the pure-Python
reference in ringfwi.fwi, so the fast C++ backend is a drop-in replacement:

    import uap as fwi   # instead of: from ringfwi import fwi

The geometry object only needs ``idx`` (grid index tuples), ``n_elements``, and
``element_index(k)``, which both RingArray and CylinderArray provide.
"""

from __future__ import annotations

import numpy as np

import _uap


def linear_indices(idx, shape):
    """Row-major linear indices for a list of grid index tuples."""
    strides = np.cumprod((1,) + tuple(shape[::-1][:-1]))[::-1]
    return np.array([int(np.dot(np.asarray(t), strides)) for t in idx], dtype=np.int32)


def _tx_lin(geom, shape, src_list):
    rec_lin = linear_indices(geom.idx, shape)
    if src_list is None:
        return rec_lin, rec_lin.copy()
    return rec_lin, np.array([rec_lin[s] for s in src_list], dtype=np.int32)


# The sponge argument is accepted (and ignored) so these are signature-compatible
# drop-ins for ringfwi.fwi; the C++ core uses the exact sponge-free scheme.
def forward_fmc(m, geom, wavelet, dt, h, nt, sponge=None, src_list=None):
    """Full-matrix-capture forward modelling on the C++ core."""
    rec_lin, tx_lin = _tx_lin(geom, m.shape, src_list)
    return _uap.forward_fmc(np.ascontiguousarray(m, float), float(h), float(dt),
                            int(nt), tx_lin, rec_lin, np.ascontiguousarray(wavelet, float))


_MISFIT_CODES = {"l2": 0, "gcn": 1}


def misfit_and_gradient(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
                        misfit_type="l2"):
    """Waveform misfit (l2 or gcn) and adjoint-state gradient on the C++ core."""
    rec_lin, tx_lin = _tx_lin(geom, m.shape, src_list)
    J, g = _uap.misfit_and_gradient(np.ascontiguousarray(m, float), float(h), float(dt),
                                    int(nt), tx_lin, rec_lin,
                                    np.ascontiguousarray(wavelet, float),
                                    np.ascontiguousarray(dobs, float),
                                    _MISFIT_CODES[misfit_type])
    return J, g


def misfit(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
           misfit_type="l2"):
    """Waveform misfit only (l2 or gcn), using the C++ forward."""
    dsyn = forward_fmc(m, geom, wavelet, dt, h, nt, src_list=src_list)
    if misfit_type == "l2":
        r = dsyn - dobs
        return 0.5 * float(np.sum(r * r))
    eps = 1e-12
    ns = np.sqrt(np.sum(dsyn * dsyn, axis=1)) + eps      # per-trace over time
    no = np.sqrt(np.sum(dobs * dobs, axis=1)) + eps
    c = np.sum((dsyn / ns[:, None]) * (dobs / no[:, None]), axis=1)
    return float(np.sum(1.0 - c))
