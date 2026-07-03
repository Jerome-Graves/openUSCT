"""3D anisotropic elastic wave propagation (velocity-stress staggered grid).

Full elastodynamics with a general 21-component stiffness tensor, so an
arbitrarily oriented anisotropic crystal (for example ice Ih with any 3D c-axis)
is propagated with its complete physics: quasi-P, both quasi-S waves, and all
mode conversion. This is the 3D extension of the verified 2D solvers in
:mod:`ringfwi.elastic` / :mod:`ringfwi.anisotropy`.

Grid (axis order (z, y, x), offsets in half cells):
    vx (0,0,+)  vy (0,+,0)  vz (+,0,0)
    sxx, syy, szz at cell centres (0,0,0)
    syz (+,+,0)  sxz (+,0,+)  sxy (0,+,+)

Stress rates follow sigma_I = C_IJ gamma_J in Voigt notation with engineering
shear strains gamma = (exx, eyy, ezz, 2eyz, 2exz, 2exy). Each strain component
lives at its natural staggered position; off-diagonal stiffness couplings move
strains between positions with half-cell averaging (the same treatment the 2D
monoclinic C16/C26 terms use).
"""

from __future__ import annotations

import numpy as np

# Half-cell offsets (z, y, x) of each staggered position.
_POS = {"c": (0, 0, 0), "yz": (1, 1, 0), "xz": (1, 0, 1), "xy": (0, 1, 1)}
# Voigt index -> position of that stress/strain component.
_VPOS = {1: "c", 2: "c", 3: "c", 4: "yz", 5: "xz", 6: "xy"}
_AX = {"z": -3, "y": -2, "x": -1}


def _sl(f, ax):
    """(hi, lo) slicing tuples along ``ax`` for arrays with any lead axes."""
    hi = [slice(None)] * f.ndim; lo = [slice(None)] * f.ndim
    hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
    return tuple(hi), tuple(lo)


def _Db(f, ax, inv_h):
    """Backward difference along ``ax`` (half -> integer position)."""
    g = np.zeros_like(f)
    hi, lo = _sl(f, ax)
    g[hi] = (f[hi] - f[lo]) * inv_h
    return g


def _Df(f, ax, inv_h):
    """Forward difference along ``ax`` (integer -> half position)."""
    g = np.zeros_like(f)
    hi, lo = _sl(f, ax)
    g[lo] = (f[hi] - f[lo]) * inv_h
    return g


def _avg_axis(f, ax, d):
    """Half-cell average along ``ax``: d=+1 integer->half, d=-1 half->integer."""
    g = np.zeros_like(f)
    hi, lo = _sl(f, ax)
    if d > 0:
        g[lo] = 0.5 * (f[lo] + f[hi])
    else:
        g[hi] = 0.5 * (f[lo] + f[hi])
    return g


def _move(f, src, dst):
    """Move a field between staggered positions by half-cell averaging."""
    for ax in range(3):
        d = _POS[dst][ax] - _POS[src][ax]
        if d:
            f = _avg_axis(f, ax - 3, d)     # last three axes (batch-safe)
    return f


def _DbT(g, ax, inv_h):
    """Exact transpose of :func:`_Db`."""
    o = np.zeros_like(g)
    hi, lo = _sl(g, ax)
    q = g[hi] * inv_h
    o[hi] += q
    o[lo] -= q
    return o


def _DfT(g, ax, inv_h):
    """Exact transpose of :func:`_Df`."""
    o = np.zeros_like(g)
    hi, lo = _sl(g, ax)
    q = g[lo] * inv_h
    o[hi] += q
    o[lo] -= q
    return o


def _avg_axis_T(g, ax, d):
    """Exact transpose of :func:`_avg_axis`."""
    o = np.zeros_like(g)
    hi, lo = _sl(g, ax)
    q = 0.5 * (g[lo] if d > 0 else g[hi])
    o[lo] += q
    o[hi] += q
    return o


def _move_T(g, src, dst):
    """Exact transpose of ``_move(., src, dst)`` (maps adjoints dst -> src)."""
    for ax in (2, 1, 0):
        d = _POS[dst][ax] - _POS[src][ax]
        if d:
            g = _avg_axis_T(g, ax - 3, d)   # last three axes (batch-safe)
    return g


GRAD_KEYS = tuple(f"C{i}{j}" for i in range(1, 7) for j in range(i, 7))


def gpu_available():
    """True when CuPy and a CUDA device are usable."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def forward(C, rho, h, dt, nt, src_idx, wavelet, rec_idx,
            source="explosive", record="pressure", store=False,
            src_pts=None, rec_groups=None, device="cpu"):
    """Full 3D anisotropic elastic forward model.

    C : dict with the 21 upper-triangle Voigt stiffnesses, keys "C11".."C66"
        (engineering-strain convention); each a scalar or (nz, ny, nx) array.
    rho : (nz, ny, nx) array. Indices are (iz, iy, ix).
    source : "explosive" (into sxx+syy+szz, quasi-P) or "fx"/"fy"/"fz".
    record : "pressure" (-(sxx+syy+szz)/3), "vx", "vy" or "vz".

    Finite-aperture elements: ``src_pts`` (list of (idx, weight)) spreads the
    transmit over an element footprint; ``rec_groups`` (list over receivers of
    (idx_list, weights)) records the weighted-average field over each footprint.

    device : "cpu" (float64 NumPy, the verified reference), "gpu" (float32
    CuPy) or "auto" (GPU when available). The wavefield history (``store``)
    stays on the CPU path.
    """
    use_gpu = False
    if device in ("auto", "gpu") and not store:
        big_enough = device == "gpu" or np.asarray(rho).size >= 40000
        if gpu_available() and big_enough:
            use_gpu = True                  # below ~40k cells launch overhead
        elif device == "gpu":               # loses to the CPU, so "auto" stays
            raise RuntimeError("device='gpu' requested but CuPy/CUDA unavailable")
    if use_gpu:
        import cupy as xp
        dtype = xp.float32
    else:
        xp = np
        dtype = np.float64

    rho_h = np.asarray(rho, float)
    nz, ny, nx = rho_h.shape
    inv_h = 1.0 / h

    # Activity decided on the host arrays (avoids device syncs per pair).
    def cij_host(i, j):
        return C[f"C{min(i, j)}{max(i, j)}"]

    active = {(I, J): bool(np.any(cij_host(I, J))) for I in range(1, 7)
              for J in range(1, 7)}
    Cd = {k: (xp.asarray(v, dtype) if np.ndim(v) else float(v))
          for k, v in C.items()}

    def cij(i, j):
        return Cd[f"C{min(i, j)}{max(i, j)}"]

    rho = xp.asarray(rho_h, dtype)
    if src_pts is None:
        src_pts = [(src_idx, 1.0)]

    def _lin(t):
        return (t[0] * ny + t[1]) * nx + t[2]

    # Vectorised receiver sampling (per-element reads would sync the GPU).
    if rec_groups is not None:
        kmax = max(len(idxs) for idxs, _ in rec_groups)
        idx_mat = np.zeros((len(rec_groups), kmax), dtype=np.int64)
        w_mat = np.zeros((len(rec_groups), kmax))
        for j, (idxs, ws) in enumerate(rec_groups):
            for k_, (ix, wi) in enumerate(zip(idxs, ws)):
                idx_mat[j, k_] = _lin(ix)
                w_mat[j, k_] = wi
        idx_mat = xp.asarray(idx_mat)
        w_mat = xp.asarray(w_mat, dtype)
        n_rx = len(rec_groups)
    else:
        rec_lin = xp.asarray(np.array([_lin(t) for t in rec_idx], dtype=np.int64))
        n_rx = len(rec_idx)

    vx = xp.zeros((nz, ny, nx), dtype); vy = xp.zeros((nz, ny, nx), dtype)
    vz = xp.zeros((nz, ny, nx), dtype)
    s = {I: xp.zeros((nz, ny, nx), dtype) for I in range(1, 7)}
    rec = xp.zeros((nt, n_rx), dtype)
    hist = None if not store else np.zeros((nt, nz, ny, nx))
    Z, Y, X = _AX["z"], _AX["y"], _AX["x"]

    for n in range(nt):
        # --- strain rates at their native positions (engineering shears) ---
        gam = {1: _Db(vx, X, inv_h), 2: _Db(vy, Y, inv_h), 3: _Db(vz, Z, inv_h),
               4: _Df(vz, Y, inv_h) + _Df(vy, Z, inv_h),
               5: _Df(vz, X, inv_h) + _Df(vx, Z, inv_h),
               6: _Df(vy, X, inv_h) + _Df(vx, Y, inv_h)}

        moved = {}

        def strain_at(J, pos):
            key = (J, pos)
            if key not in moved:
                moved[key] = (gam[J] if _VPOS[J] == pos
                              else _move(gam[J], _VPOS[J], pos))
            return moved[key]

        # --- stress update: sigma_I += dt * C_IJ gamma_J ---
        for I in range(1, 7):
            pos = _VPOS[I]
            rate = None
            for J in range(1, 7):
                if not active[(I, J)]:
                    continue
                term = cij(I, J) * strain_at(J, pos)
                rate = term if rate is None else rate + term
            if rate is not None:
                s[I] += dt * rate

        if source == "explosive":
            for idx, w in src_pts:
                for I in (1, 2, 3):
                    s[I][idx] += wavelet[n] * w

        # --- velocity update ---
        vx += (dt / rho) * (_Df(s[1], X, inv_h) + _Db(s[6], Y, inv_h) + _Db(s[5], Z, inv_h))
        vy += (dt / rho) * (_Db(s[6], X, inv_h) + _Df(s[2], Y, inv_h) + _Db(s[4], Z, inv_h))
        vz += (dt / rho) * (_Db(s[5], X, inv_h) + _Db(s[4], Y, inv_h) + _Df(s[3], Z, inv_h))

        if source == "fx":
            vx[src_idx] += wavelet[n] * dt
        elif source == "fy":
            vy[src_idx] += wavelet[n] * dt
        elif source == "fz":
            vz[src_idx] += wavelet[n] * dt

        if record == "pressure":
            field = -(s[1] + s[2] + s[3]) / 3.0
        else:
            field = {"vx": vx, "vy": vy, "vz": vz}[record]
        flat = field.ravel()
        if rec_groups is not None:
            rec[n] = (flat[idx_mat] * w_mat).sum(axis=1)
        else:
            rec[n] = flat[rec_lin]
        if hist is not None:
            hist[n] = np.sqrt(vx * vx + vy * vy + vz * vz)

    if use_gpu:
        import cupy as cp
        rec = cp.asnumpy(rec).astype(np.float64)
    return rec, hist


# ---------------------------------------------------------------------------
# Exact discrete adjoint: gradient of the data misfit with respect to all 21
# stiffness maps, by reverse-mode differentiation of the forward above. The
# same construction as the verified 2D full-stiffness adjoint in
# ringfwi.anisotropy (see tests/test_gradient_elastic3d.py for the
# finite-difference verification).
# ---------------------------------------------------------------------------

def _grad_forward(C, rho, h, dt, nt, src_idx, wavelet, rec_idx,
                  src_pts=None, rec_groups=None):
    """Explosive-source / pressure-record forward storing velocity history.

    Identical physics to :func:`forward` (CPU float64 path) but keeps
    vx/vy/vz at the start of every step -- the state the adjoint needs to
    rebuild the strain rates in reverse.
    """
    rho = np.asarray(rho, float)
    nz, ny, nx = rho.shape
    inv_h = 1.0 / h

    def cij(i, j):
        return C[f"C{min(i, j)}{max(i, j)}"]

    active = {(I, J): bool(np.any(cij(I, J))) for I in range(1, 7)
              for J in range(1, 7)}
    if src_pts is None:
        src_pts = [(src_idx, 1.0)]
    n_rx = len(rec_groups) if rec_groups is not None else len(rec_idx)

    vx = np.zeros((nz, ny, nx)); vy = np.zeros((nz, ny, nx))
    vz = np.zeros((nz, ny, nx))
    s = {I: np.zeros((nz, ny, nx)) for I in range(1, 7)}
    rec = np.zeros((nt, n_rx))
    vh = np.zeros((nt, 3, nz, ny, nx))
    Z, Y, X = _AX["z"], _AX["y"], _AX["x"]

    for n in range(nt):
        vh[n, 0] = vx; vh[n, 1] = vy; vh[n, 2] = vz
        gam = {1: _Db(vx, X, inv_h), 2: _Db(vy, Y, inv_h), 3: _Db(vz, Z, inv_h),
               4: _Df(vz, Y, inv_h) + _Df(vy, Z, inv_h),
               5: _Df(vz, X, inv_h) + _Df(vx, Z, inv_h),
               6: _Df(vy, X, inv_h) + _Df(vx, Y, inv_h)}
        moved = {}

        def strain_at(J, pos):
            key = (J, pos)
            if key not in moved:
                moved[key] = (gam[J] if _VPOS[J] == pos
                              else _move(gam[J], _VPOS[J], pos))
            return moved[key]

        for I in range(1, 7):
            pos = _VPOS[I]
            rate = None
            for J in range(1, 7):
                if not active[(I, J)]:
                    continue
                term = cij(I, J) * strain_at(J, pos)
                rate = term if rate is None else rate + term
            if rate is not None:
                s[I] += dt * rate

        for idx, w in src_pts:
            for I in (1, 2, 3):
                s[I][idx] += wavelet[n] * w

        vx += (dt / rho) * (_Df(s[1], X, inv_h) + _Db(s[6], Y, inv_h) + _Db(s[5], Z, inv_h))
        vy += (dt / rho) * (_Db(s[6], X, inv_h) + _Df(s[2], Y, inv_h) + _Db(s[4], Z, inv_h))
        vz += (dt / rho) * (_Db(s[5], X, inv_h) + _Db(s[4], Y, inv_h) + _Df(s[3], Z, inv_h))

        field = -(s[1] + s[2] + s[3]) / 3.0
        if rec_groups is not None:
            for j, (idxs, ws) in enumerate(rec_groups):
                rec[n, j] = sum(wi * field[ix] for ix, wi in zip(idxs, ws))
        else:
            for j, idx in enumerate(rec_idx):
                rec[n, j] = field[idx]
    return rec, vh


def misfit_and_gradient(C, rho, h, dt, nt, src_idx, wavelet, rec_idx, dobs,
                        src_pts=None, rec_groups=None, trace_weights=None,
                        grad_keys=None):
    """Data misfit J and its gradient w.r.t. the 21 stiffness maps.

    One explosive source recorded as pressure; J = 0.5 ||W (rec - dobs)||^2
    with optional ``trace_weights`` W (broadcast against (nt, n_rx)).
    Returns (J, {key: (nz, ny, nx) gradient}) for ``grad_keys`` (default: all
    21 upper-triangle keys). Exact to the discrete forward, so gradients are
    returned even for stiffnesses that are currently zero (the angle chain
    rule needs them).
    """
    rho = np.asarray(rho, float)
    nz, ny, nx = rho.shape
    inv_h = 1.0 / h
    if grad_keys is None:
        grad_keys = GRAD_KEYS

    def cij(i, j):
        return C[f"C{min(i, j)}{max(i, j)}"]

    active = {(I, J): bool(np.any(cij(I, J))) for I in range(1, 7)
              for J in range(1, 7)}
    rec, vh = _grad_forward(C, rho, h, dt, nt, src_idx, wavelet, rec_idx,
                            src_pts=src_pts, rec_groups=rec_groups)
    res = rec - dobs
    if trace_weights is not None:
        res = res * trace_weights
    J = 0.5 * float(np.sum(res * res))
    if trace_weights is not None:
        res = res * trace_weights        # adjoint source: W^T W (rec - dobs)

    lv = {a: np.zeros((nz, ny, nx)) for a in "xyz"}
    ls = {I: np.zeros((nz, ny, nx)) for I in range(1, 7)}
    g = {k: np.zeros((nz, ny, nx)) for k in grad_keys}
    dtr = dt / rho
    Z, Y, X = _AX["z"], _AX["y"], _AX["x"]

    for n in range(nt - 1, -1, -1):
        vx, vy, vz = vh[n, 0], vh[n, 1], vh[n, 2]
        gam = {1: _Db(vx, X, inv_h), 2: _Db(vy, Y, inv_h), 3: _Db(vz, Z, inv_h),
               4: _Df(vz, Y, inv_h) + _Df(vy, Z, inv_h),
               5: _Df(vz, X, inv_h) + _Df(vx, Z, inv_h),
               6: _Df(vy, X, inv_h) + _Df(vx, Y, inv_h)}
        moved = {}

        def strain_at(J, pos):
            key = (J, pos)
            if key not in moved:
                moved[key] = (gam[J] if _VPOS[J] == pos
                              else _move(gam[J], _VPOS[J], pos))
            return moved[key]

        # adjoint of the pressure recording
        if rec_groups is not None:
            for j, (idxs, ws) in enumerate(rec_groups):
                for ix, wi in zip(idxs, ws):
                    q = res[n, j] * wi / 3.0
                    ls[1][ix] -= q; ls[2][ix] -= q; ls[3][ix] -= q
        else:
            for j, idx in enumerate(rec_idx):
                q = res[n, j] / 3.0
                ls[1][idx] -= q; ls[2][idx] -= q; ls[3][idx] -= q

        # adjoint of the velocity update (deposits into the stress adjoints)
        ls[1] += _DfT(dtr * lv["x"], X, inv_h)
        ls[6] += _DbT(dtr * lv["x"], Y, inv_h) + _DbT(dtr * lv["y"], X, inv_h)
        ls[5] += _DbT(dtr * lv["x"], Z, inv_h) + _DbT(dtr * lv["z"], X, inv_h)
        ls[2] += _DfT(dtr * lv["y"], Y, inv_h)
        ls[4] += _DbT(dtr * lv["y"], Z, inv_h) + _DbT(dtr * lv["z"], Y, inv_h)
        ls[3] += _DfT(dtr * lv["z"], Z, inv_h)

        # gradient accumulation from the stress update
        for key in grad_keys:
            i, j = int(key[1]), int(key[2])
            if i == j:
                g[key] += ls[i] * dt * gam[i]
            else:
                g[key] += (ls[i] * dt * strain_at(j, _VPOS[i])
                           + ls[j] * dt * strain_at(i, _VPOS[j]))

        # adjoint of the stress update (deposits into the velocity adjoints).
        # w_J collects, at gamma_J's native position, the sensitivity routed
        # through every active C_IJ coupling (transposed half-cell averaging).
        w = {}
        for Jv in range(1, 7):
            acc = None
            for p in ("c", "yz", "xz", "xy"):
                tmp = None
                for I in range(1, 7):
                    if _VPOS[I] != p or not active[(I, Jv)]:
                        continue
                    term = dt * cij(I, Jv) * ls[I]
                    tmp = term if tmp is None else tmp + term
                if tmp is None:
                    continue
                if p != _VPOS[Jv]:
                    tmp = _move_T(tmp, _VPOS[Jv], p)
                acc = tmp if acc is None else acc + tmp
            if acc is not None:
                w[Jv] = acc

        if 1 in w:
            lv["x"] += _DbT(w[1], X, inv_h)
        if 2 in w:
            lv["y"] += _DbT(w[2], Y, inv_h)
        if 3 in w:
            lv["z"] += _DbT(w[3], Z, inv_h)
        if 4 in w:
            lv["z"] += _DfT(w[4], Y, inv_h); lv["y"] += _DfT(w[4], Z, inv_h)
        if 5 in w:
            lv["z"] += _DfT(w[5], X, inv_h); lv["x"] += _DfT(w[5], Z, inv_h)
        if 6 in w:
            lv["y"] += _DfT(w[6], X, inv_h); lv["x"] += _DfT(w[6], Y, inv_h)

    return J, g


def forward_batch(C, rho, h, dt, nt, src_pts_list, wavelet, rec_idx=None,
                  rec_groups=None, device="auto"):
    """All transmits of an FMC in ONE batched array pass (explosive/pressure).

    ``src_pts_list`` is a list over B transmits of element footprints
    [(idx, w), ...]; returns (B, nt, n_rx) float64. Batching amortises the
    per-op Python overhead of the step loop across transmits and lifts the
    effective array size over the GPU threshold (B x cells), so the CuPy
    path engages at grid sizes where a single transmit stays CPU-bound.
    Physics per transmit is identical to :func:`forward` (the operators are
    batch-transparent), so results match the sequential loop exactly on CPU.
    """
    B = len(src_pts_list)
    rho_h = np.asarray(rho, float)
    nz, ny, nx = rho_h.shape
    inv_h = 1.0 / h

    use_gpu = False
    if device in ("auto", "gpu"):
        big_enough = device == "gpu" or B * rho_h.size >= 40000
        if gpu_available() and big_enough:
            use_gpu = True
        elif device == "gpu":
            raise RuntimeError("device='gpu' requested but CuPy/CUDA unavailable")
    if use_gpu:
        import cupy as xp
        dtype = xp.float32
    elif device == "cpu-batch":              # batched math on CPU (tests)
        xp = np
        dtype = np.float64
    else:
        # A CPU batch is math-bound and beats nothing (measured 0.5x), so
        # fall back to the sequential verified forward per transmit.
        n_rx = (len(rec_groups) if rec_groups is not None else len(rec_idx))
        rec = np.zeros((B, nt, n_rx))
        for b, sp in enumerate(src_pts_list):
            rec[b], _ = forward(C, rho, h, dt, nt, None, wavelet, rec_idx,
                                src_pts=sp, rec_groups=rec_groups,
                                device="cpu")
        return rec

    def cij_host(i, j):
        return C[f"C{min(i, j)}{max(i, j)}"]

    active = {(I, J): bool(np.any(cij_host(I, J))) for I in range(1, 7)
              for J in range(1, 7)}
    Cd = {k: (xp.asarray(v, dtype) if np.ndim(v) else float(v))
          for k, v in C.items()}

    def cij(i, j):
        return Cd[f"C{min(i, j)}{max(i, j)}"]

    rho_d = xp.asarray(rho_h, dtype)

    def _lin(t):
        return (t[0] * ny + t[1]) * nx + t[2]

    if rec_groups is not None:
        kmax = max(len(idxs) for idxs, _ in rec_groups)
        idx_mat = np.zeros((len(rec_groups), kmax), dtype=np.int64)
        w_mat = np.zeros((len(rec_groups), kmax))
        for j, (idxs, ws) in enumerate(rec_groups):
            for k_, (ix, wi) in enumerate(zip(idxs, ws)):
                idx_mat[j, k_] = _lin(ix)
                w_mat[j, k_] = wi
        idx_mat = xp.asarray(idx_mat)
        w_mat = xp.asarray(w_mat, dtype)
        n_rx = len(rec_groups)
    else:
        rec_lin = xp.asarray(np.array([_lin(t) for t in rec_idx],
                                      dtype=np.int64))
        n_rx = len(rec_idx)

    # per-transmit source injection as flat scatter indices
    src_flat = []
    for b, pts in enumerate(src_pts_list):
        for idx, wgt in pts:
            src_flat.append((b * nz * ny * nx + _lin(idx), float(wgt)))
    src_idx_d = xp.asarray(np.array([i for i, _ in src_flat],
                                    dtype=np.int64))
    src_w_d = xp.asarray(np.array([w for _, w in src_flat]), dtype)

    vx = xp.zeros((B, nz, ny, nx), dtype)
    vy = xp.zeros((B, nz, ny, nx), dtype)
    vz = xp.zeros((B, nz, ny, nx), dtype)
    s = {I: xp.zeros((B, nz, ny, nx), dtype) for I in range(1, 7)}
    rec = xp.zeros((B, nt, n_rx), dtype)
    Z, Y, X = _AX["z"], _AX["y"], _AX["x"]

    for n in range(nt):
        gam = {1: _Db(vx, X, inv_h), 2: _Db(vy, Y, inv_h),
               3: _Db(vz, Z, inv_h),
               4: _Df(vz, Y, inv_h) + _Df(vy, Z, inv_h),
               5: _Df(vz, X, inv_h) + _Df(vx, Z, inv_h),
               6: _Df(vy, X, inv_h) + _Df(vx, Y, inv_h)}
        moved = {}

        def strain_at(J, pos):
            key = (J, pos)
            if key not in moved:
                moved[key] = (gam[J] if _VPOS[J] == pos
                              else _move(gam[J], _VPOS[J], pos))
            return moved[key]

        for I in range(1, 7):
            pos = _VPOS[I]
            rate = None
            for J in range(1, 7):
                if not active[(I, J)]:
                    continue
                term = cij(I, J) * strain_at(J, pos)
                rate = term if rate is None else rate + term
            if rate is not None:
                s[I] += dt * rate

        wn = wavelet[n]
        for I in (1, 2, 3):
            flat = s[I].ravel()
            flat[src_idx_d] += wn * src_w_d
        # (ravel is a view; the scatter lands in s[I])

        vx += (dt / rho_d) * (_Df(s[1], X, inv_h) + _Db(s[6], Y, inv_h)
                              + _Db(s[5], Z, inv_h))
        vy += (dt / rho_d) * (_Db(s[6], X, inv_h) + _Df(s[2], Y, inv_h)
                              + _Db(s[4], Z, inv_h))
        vz += (dt / rho_d) * (_Db(s[5], X, inv_h) + _Db(s[4], Y, inv_h)
                              + _Df(s[3], Z, inv_h))

        field = -(s[1] + s[2] + s[3]) / 3.0
        flat = field.reshape(B, -1)
        if rec_groups is not None:
            rec[:, n, :] = (flat[:, idx_mat] * w_mat).sum(axis=-1)
        else:
            rec[:, n, :] = flat[:, rec_lin]

    if use_gpu:
        import cupy as cp
        rec = cp.asnumpy(rec).astype(np.float64)
    return rec
