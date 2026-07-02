"""2D anisotropic elastic wave propagation (rotated staggered grid).

Anisotropic elastodynamics for media with a full in-plane stiffness tensor
(C11, C12, C22, C16, C26, C66). The motivating case is single-crystal ice Ih,
which is hexagonal (transversely isotropic): its P-wave speed depends on the
propagation direction relative to the c-axis. Resolving that directional
velocity anisotropy is the physical basis of crystal-orientation-fabric (COF)
estimation in ice cores.

The solver extends the verified isotropic velocity-stress staggered grid
(see elastic.py). Diagonal anisotropy (C11 != C22, plus C12 and C66) is captured
with no interpolation: those terms only need dvx/dx and dvy/dy, which live at the
cell centre where sxx/syy do. An arbitrary c-axis orientation additionally makes
C16/C26 non-zero (monoclinic in-plane); those couple the normal stresses to the
shear strain (and vice versa), which lives on the staggered corner, so they are
handled by a 4-point average between the centre and corner grids.

The analytical Christoffel equation Gamma_ik = C_ijkl n_j n_l gives the exact
phase velocities used to verify the solver.
"""

from __future__ import annotations

import numpy as np

# Single-crystal ice Ih stiffness (GPa) and density (kg/m^3), ~ -16 C.
# Hexagonal symmetry axis (c-axis) = x3. (Gammon et al. 1983 / Gagnon et al. 1988.)
ICE_C11 = 13.93e9
ICE_C33 = 15.01e9
ICE_C44 = 3.01e9
ICE_C12 = 7.08e9
ICE_C13 = 5.77e9
ICE_RHO = 917.0

_VOIGT = ((0, 0), (1, 1), (0, 1))   # 2D Voigt index -> tensor index pair


def voigt_to_tensor(C):
    """3x3 Voigt stiffness (engineering strain) -> 2x2x2x2 tensor."""
    T = np.zeros((2, 2, 2, 2))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, l) in enumerate(_VOIGT):
            v = C[a, b]
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    T[ii, jj, kk, ll] = v
    return T


def tensor_to_voigt(T):
    """2x2x2x2 tensor -> 3x3 Voigt stiffness."""
    C = np.zeros((3, 3))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, l) in enumerate(_VOIGT):
            C[a, b] = T[i, j, k, l]
    return C


def rotate_voigt(C, theta):
    """Rotate a 3x3 Voigt stiffness by angle theta (radians) in the plane."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    T = voigt_to_tensor(C)
    Tr = np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, T)
    return tensor_to_voigt(Tr)


def ice_stiffness_2d(theta=0.0):
    """In-plane Voigt stiffness for ice Ih in the plane containing the c-axis.

    With theta = 0 the c-axis lies along the 2D-y axis (orthotropic). Rotating by
    theta turns the c-axis in the imaging plane and introduces C16/C26.
    Mapping: 2D-x = crystal x1, 2D-y = crystal x3 (c-axis); shear = crystal 1-3
    plane (C55 = C44).
    """
    C = np.array([
        [ICE_C11, ICE_C13, 0.0],
        [ICE_C13, ICE_C33, 0.0],
        [0.0,     0.0,     ICE_C44],
    ])
    return rotate_voigt(C, theta) if theta else C


def christoffel_velocities(C, rho, phi):
    """Phase velocities (qP, qS) for propagation direction phi (radians).

    Returns (v_qP, v_qS) with v_qP >= v_qS, from the eigenvalues of the 2D
    Christoffel matrix Gamma_ik = C_ijkl n_j n_l, rho v^2 = eig(Gamma).
    """
    T = voigt_to_tensor(np.asarray(C, float))
    n = np.array([np.cos(phi), np.sin(phi)])
    Gamma = np.einsum("ijkl,j,l->ik", T, n, n)
    ev = np.sort(np.linalg.eigvalsh(Gamma))[::-1]
    return np.sqrt(ev[0] / rho), np.sqrt(ev[1] / rho)


def polycrystal_stiffness(labels, angles, c_couplant=1480.0, rho_couplant=1000.0,
                          fluid_mask=None):
    """Stiffness and density maps for an ice polycrystal in a fluid couplant.

    Each grain (``labels`` >= 0, from :func:`ringfwi.phantom.voronoi_polycrystal`)
    gets the single-crystal ice Ih stiffness rotated by its c-axis ``angle``.
    Cells outside the specimen are the fluid couplant, expressed in the same
    velocity-stress framework as a solid with zero shear stiffness:
    C11 = C22 = C12 = rho c^2 (bulk modulus), C16 = C26 = C66 = 0 (the acoustic
    limit). ``fluid_mask`` optionally forces extra cells to fluid, modelling an
    isotropic pocket inside the polycrystal (for example a melt inclusion).

    Returns (C11, C12, C22, C16, C26, C66), rho — ready for :func:`forward`.
    """
    shape = labels.shape
    K = rho_couplant * c_couplant * c_couplant
    C11 = np.full(shape, K); C12 = np.full(shape, K); C22 = np.full(shape, K)
    C16 = np.zeros(shape); C26 = np.zeros(shape); C66 = np.zeros(shape)
    rho = np.full(shape, rho_couplant)

    base = ice_stiffness_2d(0.0)
    for k, th in enumerate(np.asarray(angles, float)):
        msk = labels == k
        if not msk.any():
            continue
        Ck = rotate_voigt(base, th)
        C11[msk] = Ck[0, 0]; C12[msk] = Ck[0, 1]; C22[msk] = Ck[1, 1]
        C16[msk] = Ck[0, 2]; C26[msk] = Ck[1, 2]; C66[msk] = Ck[2, 2]
        rho[msk] = ICE_RHO

    if fluid_mask is not None:
        m = fluid_mask.astype(bool)
        C11[m] = K; C12[m] = K; C22[m] = K
        C16[m] = 0.0; C26[m] = 0.0; C66[m] = 0.0
        rho[m] = rho_couplant

    return (C11, C12, C22, C16, C26, C66), rho


def polycrystal_apparent_speed(labels, angles, c_couplant=1480.0, phi=0.0,
                               fluid_mask=None):
    """Per-grain apparent qP speed along direction ``phi`` (display / FWI ref).

    A polycrystal of identical crystals differing only in orientation has no
    isotropic velocity contrast; what a fixed propagation direction sees is the
    directional qP speed of each grain. This is the "apparent velocity" field
    an acoustic reconstruction of anisotropic data is probing.
    """
    base = ice_stiffness_2d(0.0)
    capp = np.full(labels.shape, float(c_couplant))
    for k, th in enumerate(np.asarray(angles, float)):
        msk = labels == k
        if not msk.any():
            continue
        vqp, _ = christoffel_velocities(rotate_voigt(base, th), ICE_RHO, phi)
        capp[msk] = vqp
    if fluid_mask is not None:
        capp[fluid_mask.astype(bool)] = c_couplant
    return capp


def ice_qp_vs_caxis(psi):
    """qP phase speed at angle ``psi`` between propagation and the c-axis.

    Ice Ih is transversely isotropic, so qP depends only on this angle; the
    exact value comes from the 2D Christoffel solution in the plane containing
    the c-axis (where the 2D convention has the c-axis along y, so a
    propagation angle phi from x gives psi = pi/2 - phi).
    """
    vqp, _ = christoffel_velocities(ice_stiffness_2d(0.0), ICE_RHO,
                                    np.pi / 2.0 - psi)
    return vqp


def polycrystal_apparent_speed_3d(labels, axes, c_couplant=1480.0,
                                  prop_dir=(1.0, 0.0, 0.0), fluid_mask=None,
                                  material=None):
    """Per-grain apparent qP speed map for a 3D polycrystal.

    Each grain's scalar is the exact transversely-isotropic qP phase speed of
    ``material`` (defaults to ice Ih) for propagation along ``prop_dir`` given
    the grain's 3D c-axis. Used as the display / reconstruction reference for
    the full elastic simulation.
    """
    mat = ICE_MATERIAL if material is None else material
    base6 = ti_stiffness_6(**mat)
    d = np.asarray(prop_dir, float)
    d = d / np.linalg.norm(d)
    capp = np.full(labels.shape, float(c_couplant))
    for k, ax in enumerate(np.asarray(axes, float)):
        msk = labels == k
        if not msk.any():
            continue
        psi = np.arccos(np.clip(abs(float(np.dot(ax, d))), 0.0, 1.0))
        vqp, _, _ = christoffel_3d(base6, mat["rho"],
                                   (np.sin(psi), 0.0, np.cos(psi)))
        capp[msk] = vqp
    if fluid_mask is not None:
        capp[fluid_mask.astype(bool)] = c_couplant
    return capp


# --- Full 3D stiffness machinery ---------------------------------------------

_VOIGT3 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt6_to_tensor(C):
    """6x6 Voigt stiffness (engineering strain) -> 3x3x3x3 tensor."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT3):
        for b, (k, l) in enumerate(_VOIGT3):
            v = C[a, b]
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    T[ii, jj, kk, ll] = v
    return T


def tensor_to_voigt6(T):
    """3x3x3x3 tensor -> 6x6 Voigt stiffness."""
    C = np.zeros((6, 6))
    for a, (i, j) in enumerate(_VOIGT3):
        for b, (k, l) in enumerate(_VOIGT3):
            C[a, b] = T[i, j, k, l]
    return C


def rotate_voigt_3d(C, R):
    """Rotate a 6x6 Voigt stiffness by the 3x3 rotation matrix ``R``."""
    T = voigt6_to_tensor(np.asarray(C, float))
    Tr = np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, T)
    return tensor_to_voigt6(Tr)


def _rotation_z_to(axis):
    """Rotation matrix taking the z axis onto the unit vector ``axis``."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    c = float(np.dot(z, a))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)         # Rodrigues


# Registry of common transversely isotropic (hexagonal) materials.
# Typical literature single-crystal constants (Pa; density kg/m^3); the five
# independent TI constants are C11, C33, C44, C12, C13 with C66 = (C11-C12)/2
# and the symmetry (c) axis along z.
TI_MATERIALS = {
    "Ice Ih (-16 C)": dict(C11=13.93e9, C33=15.01e9, C44=3.01e9,
                           C12=7.08e9, C13=5.77e9, rho=917.0),
    "Titanium (alpha)": dict(C11=162.4e9, C33=180.7e9, C44=46.7e9,
                             C12=92.0e9, C13=69.0e9, rho=4506.0),
    "Zinc": dict(C11=161.0e9, C33=61.0e9, C44=38.3e9,
                 C12=34.2e9, C13=50.1e9, rho=7140.0),
    "Magnesium": dict(C11=59.7e9, C33=61.7e9, C44=16.4e9,
                      C12=26.2e9, C13=21.7e9, rho=1738.0),
    "Zirconium (alpha)": dict(C11=143.4e9, C33=164.8e9, C44=32.0e9,
                              C12=72.8e9, C13=65.3e9, rho=6511.0),
    "Graphite (pyrolytic)": dict(C11=1060.0e9, C33=36.5e9, C44=4.5e9,
                                 C12=180.0e9, C13=15.0e9, rho=2266.0),
}
ICE_MATERIAL = TI_MATERIALS["Ice Ih (-16 C)"]


def ti_stiffness_6(C11, C33, C44, C12, C13, **_):
    """6x6 Voigt stiffness of a transversely isotropic solid, c-axis along z."""
    return np.array([
        [C11, C12, C13, 0, 0, 0],
        [C12, C11, C13, 0, 0, 0],
        [C13, C13, C33, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, (C11 - C12) / 2.0],
    ])


def ti_stiffness_3d(axis, material):
    """Full 6x6 stiffness of a TI ``material`` with the c-axis along ``axis``."""
    return rotate_voigt_3d(ti_stiffness_6(**material), _rotation_z_to(axis))


def ti_max_speed(material, n_psi=181):
    """Maximum phase speed of a TI material over all directions (for CFL)."""
    base = ti_stiffness_6(**material)
    vmax = 0.0
    for psi in np.linspace(0.0, np.pi / 2.0, n_psi):
        v = christoffel_3d(base, material["rho"], (np.sin(psi), 0.0, np.cos(psi)))
        vmax = max(vmax, v[0])
    return vmax


def ice_stiffness_3d(axis=(0.0, 0.0, 1.0)):
    """Full 6x6 stiffness of single-crystal ice Ih with the c-axis along ``axis``."""
    return ti_stiffness_3d(axis, ICE_MATERIAL)


def christoffel_3d(C6, rho, n):
    """Phase velocities (qP, qS1, qS2) for propagation direction ``n`` (3D)."""
    T = voigt6_to_tensor(np.asarray(C6, float))
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    Gamma = np.einsum("ijkl,j,l->ik", T, n, n)
    ev = np.sort(np.linalg.eigvalsh(Gamma))[::-1]
    return tuple(np.sqrt(e / rho) for e in ev)


_KEYS21 = [f"C{i}{j}" for i in range(1, 7) for j in range(i, 7)]


def polycrystal_stiffness_3d(labels, axes, c_couplant=1480.0, rho_couplant=1000.0,
                             fluid_mask=None, material=None):
    """Full 21-component stiffness + density maps for a 3D polycrystal.

    Each grain gets the complete rotated single-crystal tensor of ``material``
    (a dict with C11, C33, C44, C12, C13, rho — see :data:`TI_MATERIALS`;
    defaults to ice Ih) for its 3D c-axis; the couplant is fluid (bulk stiffness
    only, zero shear). Returns (Cmaps, rho) with ``Cmaps`` a dict of the 21
    upper-triangle Voigt maps, ready for :func:`ringfwi.elastic3d.forward`.
    """
    mat = ICE_MATERIAL if material is None else material
    shape = labels.shape
    K = rho_couplant * c_couplant * c_couplant
    fluid6 = np.zeros((6, 6))
    fluid6[:3, :3] = K
    Cmaps = {}
    for key in _KEYS21:
        i, j = int(key[1]) - 1, int(key[2]) - 1
        Cmaps[key] = np.full(shape, fluid6[i, j])
    rho = np.full(shape, rho_couplant)

    for k, ax in enumerate(np.asarray(axes, float)):
        msk = labels == k
        if not msk.any():
            continue
        C6 = ti_stiffness_3d(ax, mat)
        for key in _KEYS21:
            i, j = int(key[1]) - 1, int(key[2]) - 1
            Cmaps[key][msk] = C6[i, j]
        rho[msk] = mat["rho"]

    if fluid_mask is not None:
        m = fluid_mask.astype(bool)
        for key in _KEYS21:
            i, j = int(key[1]) - 1, int(key[2]) - 1
            Cmaps[key][m] = fluid6[i, j]
        rho[m] = rho_couplant

    return Cmaps, rho


def _corner_to_centre(g):
    """Average a corner-grid field (sxy location) onto the centre grid."""
    out = np.zeros_like(g)
    out[1:, 1:] = 0.25 * (g[1:, 1:] + g[:-1, 1:] + g[1:, :-1] + g[:-1, :-1])
    return out


def _centre_to_corner(f):
    """Average a centre-grid field (sxx location) onto the corner grid."""
    out = np.zeros_like(f)
    out[:-1, :-1] = 0.25 * (f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:])
    return out


def forward(C, rho, h, dt, nt, src_idx, wavelet, rec_idx,
            source="explosive", record="pressure", store=False):
    """Anisotropic elastic forward model on a velocity-stress staggered grid.

    C : dict or sequence (C11, C12, C22, C16, C26, C66); each entry a scalar or
        an (ny, nx) array. rho : scalar or (ny, nx) array.
    Sources/receivers are (iy, ix). "explosive" injects into sxx+syy (P);
    "fx"/"fy" inject a body force. record: "pressure", "vx" or "vy".
    """
    if isinstance(C, dict):
        C11, C12, C22, C16, C26, C66 = (C["C11"], C["C12"], C["C22"],
                                        C["C16"], C["C26"], C["C66"])
    else:
        C11, C12, C22, C16, C26, C66 = C

    rho_arr = np.asarray(rho, float)
    ny, nx = rho_arr.shape
    inv_h = 1.0 / h
    mono = np.any(C16) or np.any(C26)      # off-diagonal (rotated c-axis) present

    vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
    sxx = np.zeros((ny, nx)); syy = np.zeros((ny, nx)); sxy = np.zeros((ny, nx))
    rec = np.zeros((nt, len(rec_idx)))
    hist = None if not store else np.zeros((nt, ny, nx))

    for n in range(nt):
        # --- stress update ---
        dvx_dx = np.zeros((ny, nx)); dvx_dx[:, 1:] = (vx[:, 1:] - vx[:, :-1]) * inv_h  # centre
        dvy_dy = np.zeros((ny, nx)); dvy_dy[1:, :] = (vy[1:, :] - vy[:-1, :]) * inv_h  # centre
        dvy_dx = np.zeros((ny, nx)); dvy_dx[:, :-1] = (vy[:, 1:] - vy[:, :-1]) * inv_h  # corner
        dvx_dy = np.zeros((ny, nx)); dvx_dy[:-1, :] = (vx[1:, :] - vx[:-1, :]) * inv_h  # corner
        gxy_c = dvy_dx + dvx_dy                                   # shear strain at corner

        sxx_rate = C11 * dvx_dx + C12 * dvy_dy
        syy_rate = C12 * dvx_dx + C22 * dvy_dy
        sxy_rate = C66 * gxy_c
        if mono:
            gxy_centre = _corner_to_centre(gxy_c)                 # shear strain at centre
            sxx_rate = sxx_rate + C16 * gxy_centre
            syy_rate = syy_rate + C26 * gxy_centre
            sxy_rate = sxy_rate + (C16 * _centre_to_corner(dvx_dx)
                                   + C26 * _centre_to_corner(dvy_dy))
        sxx += dt * sxx_rate; syy += dt * syy_rate; sxy += dt * sxy_rate

        if source == "explosive":
            sxx[src_idx] += wavelet[n]; syy[src_idx] += wavelet[n]

        # --- velocity update ---
        dsxx_dx = np.zeros((ny, nx)); dsxx_dx[:, :-1] = (sxx[:, 1:] - sxx[:, :-1]) * inv_h
        dsxy_dy = np.zeros((ny, nx)); dsxy_dy[1:, :] = (sxy[1:, :] - sxy[:-1, :]) * inv_h
        vx += (dt / rho_arr) * (dsxx_dx + dsxy_dy)
        dsxy_dx = np.zeros((ny, nx)); dsxy_dx[:, 1:] = (sxy[:, 1:] - sxy[:, :-1]) * inv_h
        dsyy_dy = np.zeros((ny, nx)); dsyy_dy[:-1, :] = (syy[1:, :] - syy[:-1, :]) * inv_h
        vy += (dt / rho_arr) * (dsxy_dx + dsyy_dy)

        if source == "fx":
            vx[src_idx] += wavelet[n] * dt
        elif source == "fy":
            vy[src_idx] += wavelet[n] * dt

        if record == "pressure":
            field = -(sxx + syy) / 2.0
        else:
            field = vx if record == "vx" else vy
        for j, idx in enumerate(rec_idx):
            rec[n, j] = field[idx]
        if hist is not None:
            hist[n] = np.sqrt(vx * vx + vy * vy)

    return rec, hist


# ---------------------------------------------------------------------------
# Elastic FWI gradient (exact discrete adjoint).
#
# The gradient of the data misfit with respect to the stiffness maps is obtained
# by reverse-mode differentiation of the exact discrete forward, so it matches a
# finite-difference check to machine precision. Only the diagonal stiffnesses
# (C11, C12, C22, C66) are differentiated here -- the on-axis / orthotropic case,
# which already covers isotropic and c-axis-aligned ice. The off-diagonal
# C16/C26 gradient (rotated c-axis) is a follow-on.
#
# Forward difference operators and their exact transposes (adjoints). "b"/"f"
# denote the backward/forward one-sided stencils used in the forward solver.
# ---------------------------------------------------------------------------

def _Dbx(f, inv_h):
    o = np.zeros_like(f); o[:, 1:] = (f[:, 1:] - f[:, :-1]) * inv_h; return o


def _Dby(f, inv_h):
    o = np.zeros_like(f); o[1:, :] = (f[1:, :] - f[:-1, :]) * inv_h; return o


def _Dfx(f, inv_h):
    o = np.zeros_like(f); o[:, :-1] = (f[:, 1:] - f[:, :-1]) * inv_h; return o


def _Dfy(f, inv_h):
    o = np.zeros_like(f); o[:-1, :] = (f[1:, :] - f[:-1, :]) * inv_h; return o


def _DbxT(g, inv_h):
    gf = np.zeros_like(g); gf[:, 1:] += g[:, 1:] * inv_h; gf[:, :-1] -= g[:, 1:] * inv_h; return gf


def _DbyT(g, inv_h):
    gf = np.zeros_like(g); gf[1:, :] += g[1:, :] * inv_h; gf[:-1, :] -= g[1:, :] * inv_h; return gf


def _DfxT(g, inv_h):
    gf = np.zeros_like(g); gf[:, 1:] += g[:, :-1] * inv_h; gf[:, :-1] -= g[:, :-1] * inv_h; return gf


def _DfyT(g, inv_h):
    gf = np.zeros_like(g); gf[1:, :] += g[:-1, :] * inv_h; gf[:-1, :] -= g[:-1, :] * inv_h; return gf


def _grad_forward(C11, C12, C22, C66, rho, h, dt, nt, src_idx, wavelet, rec_idx):
    """Diagonal-stiffness forward that stores the velocity history for the adjoint."""
    ny, nx = rho.shape
    inv_h = 1.0 / h
    vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
    sxx = np.zeros((ny, nx)); syy = np.zeros((ny, nx)); sxy = np.zeros((ny, nx))
    rec = np.zeros((nt, len(rec_idx)))
    vxh = np.zeros((nt, ny, nx)); vyh = np.zeros((nt, ny, nx))
    for n in range(nt):
        vxh[n] = vx; vyh[n] = vy                      # state used by the strains
        exx = _Dbx(vx, inv_h); eyy = _Dby(vy, inv_h)
        gxy = _Dfx(vy, inv_h) + _Dfy(vx, inv_h)
        sxx += dt * (C11 * exx + C12 * eyy)
        syy += dt * (C12 * exx + C22 * eyy)
        sxy += dt * (C66 * gxy)
        sxx[src_idx] += wavelet[n]; syy[src_idx] += wavelet[n]
        vx += (dt / rho) * (_Dfx(sxx, inv_h) + _Dby(sxy, inv_h))
        vy += (dt / rho) * (_Dbx(sxy, inv_h) + _Dfy(syy, inv_h))
        field = -(sxx + syy) / 2.0
        for j, idx in enumerate(rec_idx):
            rec[n, j] = field[idx]
    return rec, vxh, vyh


def misfit_and_gradient(C11, C12, C22, C66, rho, h, dt, nt,
                        src_idx, wavelet, rec_idx, dobs):
    """Data misfit J and its gradient w.r.t. the diagonal stiffness maps.

    Returns (J, {"C11":..., "C12":..., "C22":..., "C66":...}) for one explosive
    source recorded as pressure. Verified against finite differences to machine
    precision (see tests/test_gradient_elastic.py).
    """
    ny, nx = rho.shape
    inv_h = 1.0 / h
    rec, vxh, vyh = _grad_forward(C11, C12, C22, C66, rho, h, dt, nt,
                                  src_idx, wavelet, rec_idx)
    res = rec - dobs
    J = 0.5 * float(np.sum(res * res))

    lvx = np.zeros((ny, nx)); lvy = np.zeros((ny, nx))
    lsxx = np.zeros((ny, nx)); lsyy = np.zeros((ny, nx)); lsxy = np.zeros((ny, nx))
    gC11 = np.zeros((ny, nx)); gC12 = np.zeros((ny, nx))
    gC22 = np.zeros((ny, nx)); gC66 = np.zeros((ny, nx))
    dtr = dt / rho

    for n in range(nt - 1, -1, -1):
        vx = vxh[n]; vy = vyh[n]
        exx = _Dbx(vx, inv_h); eyy = _Dby(vy, inv_h)
        gxy = _Dfx(vy, inv_h) + _Dfy(vx, inv_h)

        # adjoint of the pressure recording
        for j, idx in enumerate(rec_idx):
            lsxx[idx] -= 0.5 * res[n, j]
            lsyy[idx] -= 0.5 * res[n, j]

        # adjoint of the velocity update (deposits into the stress adjoints)
        lsxx += _DfxT(dtr * lvx, inv_h)
        lsxy += _DbyT(dtr * lvx, inv_h) + _DbxT(dtr * lvy, inv_h)
        lsyy += _DfyT(dtr * lvy, inv_h)

        # gradient accumulation from the stress update
        gC11 += lsxx * dt * exx
        gC12 += lsxx * dt * eyy + lsyy * dt * exx
        gC22 += lsyy * dt * eyy
        gC66 += lsxy * dt * gxy

        # adjoint of the stress update (deposits into the velocity adjoints)
        lvx += (_DbxT(dt * C11 * lsxx, inv_h) + _DbxT(dt * C12 * lsyy, inv_h)
                + _DfyT(dt * C66 * lsxy, inv_h))
        lvy += (_DbyT(dt * C12 * lsxx, inv_h) + _DbyT(dt * C22 * lsyy, inv_h)
                + _DfxT(dt * C66 * lsxy, inv_h))

    return J, {"C11": gC11, "C12": gC12, "C22": gC22, "C66": gC66}


def invert(C11, C12, C22, C66, rho, h, dt, nt, sources, wavelet, rec_idx,
           dobs_list, n_iter=12, steps=None, params=("C11", "C22", "C66"),
           update_mask=None, bounds=None, verbose=False):
    """Multi-source elastic FWI by normalised steepest descent.

    Inverts the stiffness maps named in ``params`` (default: the P and S
    stiffnesses C11, C22, C66; C12 held fixed to limit cross-talk). ``steps`` is
    a per-parameter relative step size. Returns the updated maps and the misfit
    history.
    """
    C = {"C11": np.array(C11, float), "C12": np.array(C12, float),
         "C22": np.array(C22, float), "C66": np.array(C66, float)}
    if steps is None:
        steps = {k: 0.05 for k in params}
    mask = 1.0 if update_mask is None else update_mask
    hist = []

    for it in range(n_iter):
        J = 0.0
        gtot = {k: np.zeros_like(C[k]) for k in params}
        for src, dobs in zip(sources, dobs_list):
            Js, g = misfit_and_gradient(C["C11"], C["C12"], C["C22"], C["C66"],
                                        rho, h, dt, nt, src, wavelet, rec_idx, dobs)
            J += Js
            for k in params:
                gtot[k] += g[k]
        for k in params:
            gmax = np.max(np.abs(gtot[k])) + 1e-30
            C[k] = C[k] - steps[k] * np.mean(C[k]) * (gtot[k] / gmax) * mask
            if bounds and k in bounds:
                C[k] = np.clip(C[k], bounds[k][0], bounds[k][1])
        hist.append(J)
        if verbose:
            print(f"  iter {it:2d}  misfit {J:.4e}")

    return C["C11"], C["C12"], C["C22"], C["C66"], hist
