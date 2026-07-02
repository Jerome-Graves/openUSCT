"""Finite-difference building blocks for the 2D acoustic forward model.

The wave equation is written in the symmetric squared-slowness form

    m(x) d2p/dt2 = laplacian(p) + f(x, t),        m(x) = 1 / c(x)**2

and integrated with a second-order-in-time, second-order-in-space leapfrog
scheme. Because the operator ``m d2/dt2 - laplacian`` is self-adjoint, the exact
discrete adjoint used by the FWI gradient (see :mod:`ringfwi.fwi`) shares this
same stencil.

This module holds the reusable pieces: the discrete Laplacian and an optional
boundary sponge. The time-stepping engine itself lives in :mod:`ringfwi.fwi`
alongside its exact adjoint so the two can never drift apart.
"""

from __future__ import annotations

import numpy as np


def make_sponge(shape, width=25, strength=0.08):
    """Return a multiplicative damping mask (values in (0, 1]).

    The mask is 1.0 in the interior and tapers smoothly toward the edges. Note
    that the current FWI gradient is the exact adjoint of the *sponge-free*
    scheme; absorbing boundaries with a matching adjoint (a split-field PML) are
    on the roadmap. The mask is provided for forward-only visualisation.
    """
    ny, nx = shape
    idx = np.arange(width)
    taper = np.exp(-((strength * (width - idx)) ** 2))

    px = np.ones(nx)
    px[:width] = taper
    px[-width:] = taper[::-1]

    py = np.ones(ny)
    py[:width] = taper
    py[-width:] = taper[::-1]

    return np.outer(py, px)


def _laplacian(p, inv_h2, out):
    """Second-order finite-difference Laplacian, written into ``out``.

    Dimension-general: a 5-point stencil in 2D, a 7-point stencil in 3D, and so
    on. Only interior cells are filled; the one-cell border stays zero.
    """
    ndim = p.ndim
    out[...] = 0.0
    inner = tuple(slice(1, -1) for _ in range(ndim))
    acc = -2.0 * ndim * p[inner]
    for ax in range(ndim):
        plus = list(inner)
        minus = list(inner)
        plus[ax] = slice(2, None)
        minus[ax] = slice(0, -2)
        acc = acc + p[tuple(plus)] + p[tuple(minus)]
    out[inner] = acc * inv_h2
    return out
