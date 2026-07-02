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
_AX = {"z": 0, "y": 1, "x": 2}


def _Db(f, ax, inv_h):
    """Backward difference along ``ax`` (half -> integer position)."""
    g = np.zeros_like(f)
    hi = [slice(None)] * 3; lo = [slice(None)] * 3
    hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
    g[tuple(hi)] = (f[tuple(hi)] - f[tuple(lo)]) * inv_h
    return g


def _Df(f, ax, inv_h):
    """Forward difference along ``ax`` (integer -> half position)."""
    g = np.zeros_like(f)
    hi = [slice(None)] * 3; lo = [slice(None)] * 3
    hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
    g[tuple(lo)] = (f[tuple(hi)] - f[tuple(lo)]) * inv_h
    return g


def _avg_axis(f, ax, d):
    """Half-cell average along ``ax``: d=+1 integer->half, d=-1 half->integer."""
    g = np.zeros_like(f)
    hi = [slice(None)] * 3; lo = [slice(None)] * 3
    hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
    if d > 0:
        g[tuple(lo)] = 0.5 * (f[tuple(lo)] + f[tuple(hi)])
    else:
        g[tuple(hi)] = 0.5 * (f[tuple(lo)] + f[tuple(hi)])
    return g


def _move(f, src, dst):
    """Move a field between staggered positions by half-cell averaging."""
    for ax in range(3):
        d = _POS[dst][ax] - _POS[src][ax]
        if d:
            f = _avg_axis(f, ax, d)
    return f


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
