"""Full waveform inversion for a ring array (2D acoustic).

The forward model is a second-order leapfrog integration of the acoustic wave
equation in squared-slowness form ``m d2p/dt2 = laplacian(p) + f`` (see the
module docstring of :mod:`ringfwi.solver`). The inverse solver is adjoint-state
full waveform inversion.

Rather than approximate the adjoint with the continuous imaging condition, the
gradient here is the *exact discrete adjoint* of the leapfrog scheme, obtained
by differentiating the discrete Lagrangian. For the update

    p^n = 2 p^{n-1} - p^{n-2} + S (L p^{n-1} + f^{n-1}),   S = dt^2 / m

the adjoint field satisfies the time-reversed recursion

    lam^k = 2 lam^{k+1} - lam^{k+2} + L (S lam^{k+1}) - R^T r^k

driven by the data residual r, and the gradient of the misfit with respect to
the squared slowness is

    g_i = (1 / m_i) * sum_n lam^n_i * (p^n - 2 p^{n-1} + p^{n-2})_i.

Because it is the exact discrete adjoint, it matches a finite-difference
gradient to machine-ish precision (see tests/test_gradient.py).
"""

from __future__ import annotations

import numpy as np

from .solver import _laplacian


def _forward(m, h, dt, nt, sources, rec_idx=None, sponge=None, store=False,
             rec_groups=None):
    """Leapfrog forward integration with clean p^n indexing.

    ``sources`` is a list of (index, series); ``index`` is a grid index tuple
    (``(iy, ix)`` in 2D, ``(iz, iy, ix)`` in 3D) and ``series`` has length
    ``nt``, the sample used when advancing to level ``n`` being ``series[n-1]``.

    Receivers are either single points (``rec_idx``) or finite-aperture groups
    (``rec_groups``, a list of ``(idx_list, weights)``); a group records the
    weighted-average pressure over its footprint.

    Works in any dimension: the array shape follows ``m.shape``.

    Returns (rec, hist) where ``rec[n]`` and ``hist[n]`` hold p at level n.
    """
    shape = m.shape
    inv_h2 = 1.0 / (h * h)
    S = (dt * dt) / m

    p_prev = np.zeros(shape)  # p^{n-2}
    p_cur = np.zeros(shape)   # p^{n-1}
    lap = np.zeros(shape)

    n_rx = (len(rec_groups) if rec_groups is not None
            else (len(rec_idx) if rec_idx is not None else 0))
    rec = None if n_rx == 0 else np.zeros((nt, n_rx))
    hist = None if not store else np.zeros((nt,) + shape, dtype=np.float64)

    for n in range(1, nt):
        _laplacian(p_cur, inv_h2, lap)
        for idx, series in sources:
            lap[idx] += series[n - 1]
        p_new = 2.0 * p_cur - p_prev + S * lap
        if sponge is not None:
            p_new *= sponge
        p_prev, p_cur = p_cur, p_new
        if hist is not None:
            hist[n] = p_cur
        if rec is not None:
            if rec_groups is not None:
                for j, (idxs, w) in enumerate(rec_groups):
                    rec[n, j] = sum(wi * p_cur[ix] for ix, wi in zip(idxs, w))
            else:
                for j, idx in enumerate(rec_idx):
                    rec[n, j] = p_cur[idx]

    return rec, hist


def _tx_sources(geom, s, wavelet, footprints):
    """Transmit sources for element ``s``: a point, or its weighted footprint."""
    if footprints is None:
        return [(geom.element_index(s), wavelet)]
    idxs, w = footprints[s]
    return [(ix, wavelet * wi) for ix, wi in zip(idxs, w)]


def simulate_wavefield(m, geom, src_element, wavelet, dt, h, nt, sponge=None):
    """Return the full pressure-field history for one transmitting element.

    Useful for visualising wave propagation through the specimen.
    """
    src_idx = geom.element_index(src_element)
    _, hist = _forward(m, h, dt, nt, [(src_idx, wavelet)], None, sponge, store=True)
    return hist


def forward_fmc(m, geom, wavelet, dt, h, nt, sponge=None, src_list=None,
                footprints=None):
    """Simulate a full-matrix-capture dataset (each element transmits in turn).

    ``footprints`` (from :func:`ringfwi.geometry.build_footprints`) models finite
    -aperture elements; when ``None`` each element is a single grid point.
    """
    if src_list is None:
        src_list = list(range(geom.n_elements))
    else:
        src_list = list(src_list)
    rec_idx = None if footprints is not None else geom.idx
    data = np.zeros((len(src_list), nt, geom.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = _forward(m, h, dt, nt, _tx_sources(geom, s, wavelet, footprints),
                          rec_idx, sponge, rec_groups=footprints)
        data[i] = rec
    return data


def _adjoint_residual(dsyn, dobs, misfit_type="l2"):
    """Misfit value and adjoint source for one transmit's data (nt, n_rx).

    - "l2":  least-squares waveform misfit; adjoint source is the residual.
    - "gcn": global correlation norm (normalised cross-correlation) per trace,
      a robust misfit that widens the convergence basin against cycle skipping.
    """
    if misfit_type == "l2":
        r = dsyn - dobs
        return 0.5 * float(np.sum(r * r)), r
    if misfit_type == "gcn":
        eps = 1e-12
        ns = np.sqrt(np.sum(dsyn * dsyn, axis=0)) + eps      # (n_rx,)
        no = np.sqrt(np.sum(dobs * dobs, axis=0)) + eps
        shat = dsyn / ns
        ohat = dobs / no
        c = np.sum(shat * ohat, axis=0)                      # correlation per trace
        J = float(np.sum(1.0 - c))
        adj = -(ohat - c * shat) / ns                        # d(1 - c)/d(dsyn)
        return J, adj
    raise ValueError(f"unknown misfit_type {misfit_type!r}")


def misfit(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
           misfit_type="l2", footprints=None, time_weights=None):
    """Waveform misfit only (a forward pass, no gradient).

    ``time_weights`` optionally windows the data: a list over transmits of
    (nt, n_rx) weight arrays (for example a P-arrival mute that removes the
    late shear/coda energy an acoustic operator cannot model).
    """
    dsyn = forward_fmc(m, geom, wavelet, dt, h, nt, sponge, src_list, footprints)
    J = 0.0
    for i in range(dsyn.shape[0]):
        if time_weights is not None:
            W = time_weights[i]
            Ji, _ = _adjoint_residual(dsyn[i] * W, dobs[i] * W, misfit_type)
        else:
            Ji, _ = _adjoint_residual(dsyn[i], dobs[i], misfit_type)
        J += Ji
    return J


def misfit_and_gradient(m, geom, wavelet, dt, h, nt, dobs, sponge=None, src_list=None,
                        misfit_type="l2", footprints=None, time_weights=None):
    """Waveform misfit and its adjoint-state gradient (``misfit_type``: l2 or gcn).

    ``footprints`` models finite-aperture elements consistently on transmit and
    receive; the adjoint source is spread over each receiver's footprint.
    ``time_weights`` windows the data per transmit (see :func:`misfit`); the
    adjoint source is weighted by the same window (exact chain rule).

    Returns
    -------
    J : float
    g : (ny, nx) ndarray
        Gradient of J with respect to the squared-slowness field.
    """
    if src_list is None:
        src_list = list(range(geom.n_elements))
    else:
        src_list = list(src_list)
    rec_idx = None if footprints is not None else geom.idx
    shape = m.shape
    inv_h2 = 1.0 / (h * h)
    S = (dt * dt) / m

    J = 0.0
    g = np.zeros(shape)
    lap = np.zeros(shape)

    for i, s in enumerate(src_list):
        # Forward pass, storing the full history.
        dsyn, U = _forward(m, h, dt, nt, _tx_sources(geom, s, wavelet, footprints),
                           rec_idx, sponge, store=True, rec_groups=footprints)
        if time_weights is not None:
            W = time_weights[i]
            Ji, res = _adjoint_residual(dsyn * W, dobs[i] * W, misfit_type)
            res = res * W                       # chain rule through the window
        else:
            Ji, res = _adjoint_residual(dsyn, dobs[i], misfit_type)   # adjoint source
        J += Ji

        # Backward adjoint recursion: lam^k = 2 lam^{k+1} - lam^{k+2}
        #                                     + L(S lam^{k+1}) - R^T r^k
        lam_p1 = np.zeros(shape)  # lam^{k+1}
        lam_p2 = np.zeros(shape)  # lam^{k+2}
        for k in range(nt - 1, 0, -1):
            _laplacian(S * lam_p1, inv_h2, lap)
            lam_k = 2.0 * lam_p1 - lam_p2 + lap
            if sponge is not None:
                lam_k *= sponge
            # inject -R^T r^k at receiver points (spread over footprints)
            if footprints is not None:
                for j, (idxs, w) in enumerate(footprints):
                    for ix, wi in zip(idxs, w):
                        lam_k[ix] -= res[k, j] * wi
            else:
                for j, idx in enumerate(rec_idx):
                    lam_k[idx] -= res[k, j]

            # Gradient accumulation: g_i += (1/m_i) lam^k (p^k - 2p^{k-1} + p^{k-2})
            d2p = U[k] - 2.0 * U[k - 1] + (U[k - 2] if k >= 2 else 0.0)
            g += lam_k * d2p

            lam_p2 = lam_p1
            lam_p1 = lam_k

    g /= m
    return J, g


def invert(
    m0,
    geom,
    wavelet,
    dt,
    h,
    nt,
    dobs,
    sponge=None,
    src_list=None,
    n_iter=15,
    step_frac=0.02,
    update_mask=None,
    m_bounds=None,
    smooth_sigma=1.0,
    n_linesearch=8,
    verbose=False,
    backend=None,
    misfit_type="l2",
    footprints=None,
    progress=None,
    time_weights=None,
):
    """Preconditioned steepest-descent FWI with a backtracking line search.

    The gradient is optionally Gaussian-smoothed (``smooth_sigma`` cells) to
    suppress the source/receiver imprint, then a backtracking line search along
    the descent direction guarantees the misfit never increases.

    ``backend`` optionally supplies faster ``misfit_and_gradient`` and ``misfit``
    implementations (for example the C++ ``uap`` module); it must be
    signature-compatible with this module. Defaults to the pure-Python core.
    ``progress`` is an optional callback called as ``progress(done, total)``
    after each completed iteration (for GUI progress reporting).

    Returns (m, history) where ``history`` is the misfit per accepted iteration.
    """
    from scipy.ndimage import gaussian_filter

    if (misfit_type == "l2" and footprints is None and time_weights is None
            and backend is not None):
        mg = backend.misfit_and_gradient
        mf = backend.misfit
    elif misfit_type == "l2" and footprints is None and time_weights is None:
        mg = misfit_and_gradient
        mf = misfit
    else:
        # Non-L2 misfits (gcn), finite-aperture footprints, and windowed data
        # use the Python reference; the C++ backend is point-element L2 only.
        mg = (lambda mm, g, w, d_, h_, n_, ob, sp, sl:
              misfit_and_gradient(mm, g, w, d_, h_, n_, ob, sp, sl, misfit_type,
                                  footprints, time_weights))
        mf = (lambda mm, g, w, d_, h_, n_, ob, sp, sl:
              misfit(mm, g, w, d_, h_, n_, ob, sp, sl, misfit_type,
                     footprints, time_weights))

    m = m0.copy()
    # Effective bounds never push a cell across its own starting value, so a
    # region that starts outside the bounds (for example a couplant bath that is
    # inside the update mask) is not corrupted by clipping.
    eff_lo = eff_hi = None
    if m_bounds is not None:
        eff_lo = np.minimum(m_bounds[0], m0)
        eff_hi = np.maximum(m_bounds[1], m0)
    history = []
    J, g = mg(m, geom, wavelet, dt, h, nt, dobs, sponge, src_list)
    for it in range(n_iter):
        history.append(J)
        if verbose:
            print(f"  iter {it:2d}  misfit = {J:.6e}")

        gm = g * update_mask if update_mask is not None else g.copy()
        if smooth_sigma:
            gm = gaussian_filter(gm, smooth_sigma)
            if update_mask is not None:
                gm = gm * update_mask
        gmax = float(np.max(np.abs(gm)))
        if gmax == 0.0:
            break

        direction = -gm
        step = step_frac * float(np.mean(m)) / gmax
        accepted = False
        for _ in range(n_linesearch):
            m_try = m + step * direction
            if m_bounds is not None:
                clipped = np.clip(m_try, eff_lo, eff_hi)
                # Only enforce bounds where the model is allowed to change, so a
                # couplant bath outside the update region is left untouched.
                m_try = np.where(update_mask > 0, clipped, m_try) if update_mask is not None else clipped
            J_try = mf(m_try, geom, wavelet, dt, h, nt, dobs, sponge, src_list)
            if J_try < J:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if verbose:
                print("  line search failed to reduce misfit; stopping")
            break

        m = m_try
        J, g = mg(m, geom, wavelet, dt, h, nt, dobs, sponge, src_list)
        if progress is not None:
            progress(it + 1, n_iter)

    history.append(J)
    return m, history


def p_window_weights(dobs, dt, f0, onset_frac=0.2, keep_cycles=2.5,
                     taper_cycles=0.75):
    """Per-trace P-arrival mute windows from the observed data.

    For each transmit/receiver trace, the onset is picked from the envelope
    (first sample above ``onset_frac`` of the trace maximum); the window keeps
    ``keep_cycles`` of f0 after the onset and cosine-tapers off over
    ``taper_cycles``. This removes the later shear/mode-converted coda that an
    acoustic operator cannot model, which otherwise dominates the misfit.

    Returns a list over transmits of (nt, n_rx) weight arrays for
    ``time_weights``.
    """
    from scipy.signal import hilbert

    n_tx, nt, n_rx = dobs.shape
    keep = keep_cycles / f0 / dt
    taper = max(1.0, taper_cycles / f0 / dt)
    t_idx = np.arange(nt)[:, None]
    weights = []
    for i in range(n_tx):
        env = np.abs(hilbert(dobs[i], axis=0))
        thresh = onset_frac * (env.max(axis=0, keepdims=True) + 1e-30)
        onset = np.argmax(env > thresh, axis=0)         # (n_rx,)
        end = onset + keep
        ramp = (end + taper - t_idx) / taper
        W = np.clip(ramp, 0.0, 1.0)
        W = 0.5 - 0.5 * np.cos(np.pi * W)               # smooth cosine roll-off
        weights.append(W)
    return weights


def fwi_from_dataset(dataset, start_c=None, update_frac=0.95, m_bounds=None,
                     n_iter=12, step_frac=0.03, verbose=False, **invert_kw):
    """Run FWI directly on a portable :class:`~ringfwi.dataset.Dataset`.

    Rebuilds a computational grid from the dataset, maps the physical element
    positions onto grid indices, and inverts. This closes the pipeline: any
    dataset (simulated or, in future, from hardware) can be reconstructed by the
    same call, so FWI and TFM are interchangeable on the same data.

    Parameters
    ----------
    dataset : Dataset
    start_c : ndarray, optional
        Starting sound-speed model on the reconstruction grid. Defaults to a
        homogeneous medium at the dataset's nominal speed.
    update_frac : float
        Update only within this fraction of the array radius.

    Returns
    -------
    dict with keys: c (recovered velocity), history, extent, h, geom.
    """
    from .geometry import GridArray
    from .phantom import velocity_to_m, m_to_velocity

    ds = dataset
    dim = ds.geometry.dim

    # Reconstruction grid: reuse the acquisition grid if the dataset carries it,
    # otherwise choose a spacing from the wavelength.
    if ds.ground_truth is not None:
        h = ds.ground_truth["h_m"]
        shape = ds.ground_truth["c"].shape
    else:
        ppw = 5.0
        h = ds.nominal_speed_m_s / (ds.tx_centre_freq_hz * ppw)
        npix = int(round(2.2 * ds.geometry.radius_m / h)) + 1
        shape = (npix,) * dim
    n = shape[0]
    domain = (n - 1) * h

    # Map centred physical element positions to grid index tuples.
    abspos = ds.geometry.element_pos + domain / 2.0
    idx = [tuple(int(round(p[a] / h)) for a in reversed(range(dim))) for p in abspos]
    geom = GridArray(idx=idx, n=n, h=h, radius_m=ds.geometry.radius_m, domain_m=domain)

    # Starting model and update region.
    if start_c is None:
        start_c = np.full(shape, ds.nominal_speed_m_s)
    m0 = velocity_to_m(start_c)

    centre = domain / 2.0
    coords = np.mgrid[tuple(slice(0, n) for _ in range(dim))].astype(float) * h
    # Radial distance in the transverse plane. In 3D the array axis is axis 0
    # (z), so the transverse plane is axes (1, 2); the update region is a
    # cylinder, full extent along the axis. In 2D both axes are transverse.
    transverse = range(dim) if dim == 2 else range(1, dim)
    r = np.sqrt(sum((coords[a] - centre) ** 2 for a in transverse))
    update_mask = (r <= ds.geometry.radius_m * update_frac).astype(float)

    m_rec, history = invert(
        m0, geom, ds.tx_wavelet, ds.dt, h, ds.n_samples, ds.data,
        src_list=list(ds.tx_elements), n_iter=n_iter, step_frac=step_frac,
        update_mask=update_mask, m_bounds=m_bounds, verbose=verbose, **invert_kw,
    )

    return {
        "c": m_to_velocity(m_rec),
        "history": history,
        "extent": (-centre, domain - centre),
        "h": h,
        "geom": geom,
    }
