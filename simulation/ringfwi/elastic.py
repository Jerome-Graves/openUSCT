"""2D elastic wave propagation (velocity-stress staggered grid).

Isotropic elastodynamics solved with the standard Virieux (1986) velocity-stress
staggered-grid scheme, which supports both compressional (P) and shear (S)
waves. This is the forward-model foundation for elastic FWI and matches the
staggered-grid elastodynamic framework used for crystal-orientation-fabric work
in ice cores.

Fields (each stored on an [iy, ix] grid, staggered in the usual way):
    vx  at (ix+1/2, iy)      vy  at (ix, iy+1/2)
    sxx, syy at (ix, iy)     sxy at (ix+1/2, iy+1/2)

Governing equations (density rho, Lame parameters lambda, mu):
    rho dvx/dt = dsxx/dx + dsxy/dy
    rho dvy/dt = dsxy/dx + dsyy/dy
    dsxx/dt = (lambda+2mu) dvx/dx + lambda dvy/dy
    dsyy/dt = lambda dvx/dx + (lambda+2mu) dvy/dy
    dsxy/dt = mu (dvy/dx + dvx/dy)

with wave speeds Vp = sqrt((lambda+2mu)/rho), Vs = sqrt(mu/rho).
"""

from __future__ import annotations

import numpy as np


def lame_from_speeds(vp, vs, rho):
    """Lame parameters from P/S speeds and density."""
    mu = rho * vs * vs
    lam = rho * vp * vp - 2.0 * mu
    return lam, mu


def forward(vp, vs, rho, h, dt, nt, src_idx, wavelet, rec_idx,
            source="explosive", record="pressure", store=False):
    """Elastic forward model.

    source : "explosive" injects into sxx+syy (pure P); "fx"/"fy" inject a body
             force into vx/vy (radiates P and S).
    record : "pressure" records -(sxx+syy)/2; "vx"/"vy" record a velocity.
    Returns (rec, hist) where hist is the pressure field history if ``store``.
    """
    ny, nx = vp.shape
    lam, mu = lame_from_speeds(vp, vs, rho)
    lam2mu = lam + 2.0 * mu
    inv_h = 1.0 / h

    vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
    sxx = np.zeros((ny, nx)); syy = np.zeros((ny, nx)); sxy = np.zeros((ny, nx))
    rec = np.zeros((nt, len(rec_idx)))
    hist = None if not store else np.zeros((nt, ny, nx))

    for n in range(nt):
        # --- stress update (uses current velocities) ---
        dvx_dx = np.zeros((ny, nx)); dvx_dx[:, 1:] = (vx[:, 1:] - vx[:, :-1]) * inv_h
        dvy_dy = np.zeros((ny, nx)); dvy_dy[1:, :] = (vy[1:, :] - vy[:-1, :]) * inv_h
        sxx += dt * (lam2mu * dvx_dx + lam * dvy_dy)
        syy += dt * (lam * dvx_dx + lam2mu * dvy_dy)
        dvy_dx = np.zeros((ny, nx)); dvy_dx[:, :-1] = (vy[:, 1:] - vy[:, :-1]) * inv_h
        dvx_dy = np.zeros((ny, nx)); dvx_dy[:-1, :] = (vx[1:, :] - vx[:-1, :]) * inv_h
        sxy += dt * mu * (dvy_dx + dvx_dy)

        if source == "explosive":
            sxx[src_idx] += wavelet[n]; syy[src_idx] += wavelet[n]

        # --- velocity update (uses new stresses) ---
        dsxx_dx = np.zeros((ny, nx)); dsxx_dx[:, :-1] = (sxx[:, 1:] - sxx[:, :-1]) * inv_h
        dsxy_dy = np.zeros((ny, nx)); dsxy_dy[1:, :] = (sxy[1:, :] - sxy[:-1, :]) * inv_h
        vx += (dt / rho) * (dsxx_dx + dsxy_dy)
        dsxy_dx = np.zeros((ny, nx)); dsxy_dx[:, 1:] = (sxy[:, 1:] - sxy[:, :-1]) * inv_h
        dsyy_dy = np.zeros((ny, nx)); dsyy_dy[:-1, :] = (syy[1:, :] - syy[:-1, :]) * inv_h
        vy += (dt / rho) * (dsxy_dx + dsyy_dy)

        if source == "fx":
            vx[src_idx] += wavelet[n] * (dt / rho[src_idx] if np.ndim(rho) else dt / rho)
        elif source == "fy":
            vy[src_idx] += wavelet[n] * (dt / rho[src_idx] if np.ndim(rho) else dt / rho)

        field = (-(sxx + syy) / 2.0 if record == "pressure"
                 else vx if record == "vx" else vy)
        for j, idx in enumerate(rec_idx):
            rec[n, j] = field[idx]
        if hist is not None:
            hist[n] = np.sqrt(vx * vx + vy * vy)   # shows both P and S fronts

    return rec, hist
