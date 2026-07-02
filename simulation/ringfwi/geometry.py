"""Ring-array acquisition geometry.

A ring array places ``n_elements`` transducers evenly around a circle of a
given radius, centred in a square imaging domain. This mirrors the
circumferential scanning geometry used for cylindrical specimens such as ice
cores, pipe sections, or laboratory tomography phantoms. In a full-matrix
capture (FMC) acquisition each element transmits in turn while every element
records, so a ring of N elements yields an N-by-N multistatic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingArray:
    """Ring of transducer elements mapped onto a finite-difference grid.

    Parameters
    ----------
    n_elements : int
        Number of transducer elements around the ring.
    radius_m : float
        Ring radius in metres.
    domain_m : float
        Side length of the square imaging domain in metres.
    h : float
        Grid spacing in metres.
    """

    n_elements: int
    radius_m: float
    domain_m: float
    h: float

    def __post_init__(self):
        self.n = int(round(self.domain_m / self.h)) + 1
        centre = self.domain_m / 2.0
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_elements, endpoint=False)
        xs = centre + self.radius_m * np.cos(angles)
        ys = centre + self.radius_m * np.sin(angles)
        self.angles = angles
        self.xy = np.column_stack([xs, ys])
        # Nearest grid index for each element (iy, ix); note row = y, col = x.
        self.idx = [
            (int(round(y / self.h)), int(round(x / self.h))) for x, y in self.xy
        ]

    @property
    def grid_shape(self):
        return (self.n, self.n)

    def element_index(self, k):
        """Grid (iy, ix) of element ``k``."""
        return self.idx[k]

    def receiver_indices(self, exclude=None):
        """All element grid points, optionally excluding one element."""
        return [ij for k, ij in enumerate(self.idx) if k != exclude]

    @property
    def element_positions(self):
        """Element (x, y) positions in metres, origin at the array centre."""
        return self.xy - self.domain_m / 2.0

    array_type = "ring"


@dataclass
class CylinderArray:
    """Cylindrical array: rings of elements stacked along the axis (3D).

    A single planar ring only images one slice. Stacking ``n_rings`` rings along
    the z axis gives an axial aperture, which is what makes true 3D
    reconstruction possible. The specimen sits inside the cylinder.

    Parameters
    ----------
    n_rings : int
        Number of rings along the axis.
    per_ring : int
        Elements per ring.
    radius_m : float
        Ring radius, metres.
    height_m : float
        Axial span covered by the outermost rings, metres.
    domain_m : float
        Side length of the cubic imaging domain, metres.
    h : float
        Grid spacing, metres.
    """

    n_rings: int
    per_ring: int
    radius_m: float
    height_m: float
    domain_m: float
    h: float

    def __post_init__(self):
        self.n = int(round(self.domain_m / self.h)) + 1
        c = self.domain_m / 2.0
        if self.n_rings == 1:
            zs = np.array([c])
        else:
            zs = c + np.linspace(-self.height_m / 2.0, self.height_m / 2.0, self.n_rings)
        ang = np.linspace(0.0, 2.0 * np.pi, self.per_ring, endpoint=False)

        xyz = []
        for z in zs:
            for a in ang:
                xyz.append([c + self.radius_m * np.cos(a),
                            c + self.radius_m * np.sin(a), z])
        self.xyz = np.asarray(xyz)  # absolute coords, metres (x, y, z)
        # Grid index per element (iz, iy, ix); axis order is (z, y, x).
        self.idx = [
            (int(round(z / self.h)), int(round(y / self.h)), int(round(x / self.h)))
            for x, y, z in self.xyz
        ]
        self.n_elements = len(self.idx)

    @property
    def grid_shape(self):
        return (self.n, self.n, self.n)

    def element_index(self, k):
        return self.idx[k]

    @property
    def element_positions(self):
        """Element (x, y, z) positions in metres, origin at the array centre."""
        return self.xyz - self.domain_m / 2.0

    array_type = "cylinder"


def _fibonacci_directions(n, hemisphere=False):
    """n near-uniform unit vectors on a sphere (or upper hemisphere)."""
    i = np.arange(n) + 0.5
    z = (1.0 - i / n) if hemisphere else (1.0 - 2.0 * i / n)   # cos(polar angle)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i                       # golden angle
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


@dataclass
class ShellArray:
    """Elements near-uniformly distributed on a spherical shell (3D).

    A full sphere gives the largest 3D aperture; a hemisphere is a one-sided
    (limited-aperture) acquisition. The specimen sits inside the shell.
    """

    n_elements: int
    radius_m: float
    domain_m: float
    h: float
    hemisphere: bool = False

    def __post_init__(self):
        self.n = int(round(self.domain_m / self.h)) + 1
        c = self.domain_m / 2.0
        dirs = _fibonacci_directions(self.n_elements, hemisphere=self.hemisphere)
        self.xyz = c + self.radius_m * dirs                    # absolute coords, metres
        self.idx = [
            (int(round(z / self.h)), int(round(y / self.h)), int(round(x / self.h)))
            for x, y, z in self.xyz
        ]

    @property
    def grid_shape(self):
        return (self.n, self.n, self.n)

    def element_index(self, k):
        return self.idx[k]

    @property
    def element_positions(self):
        return self.xyz - self.domain_m / 2.0

    @property
    def array_type(self):
        return "hemisphere" if self.hemisphere else "sphere"


def HemisphereArray(n_elements, radius_m, domain_m, h):
    """Elements on the upper hemisphere (limited-aperture 3D acquisition)."""
    return ShellArray(n_elements, radius_m, domain_m, h, hemisphere=True)


def SphereArray(n_elements, radius_m, domain_m, h):
    """Elements on a full spherical shell (maximal 3D aperture)."""
    return ShellArray(n_elements, radius_m, domain_m, h, hemisphere=False)


def _tangent_basis(normal):
    """One (2D) or two (3D) unit vectors spanning the plane normal to ``normal``."""
    normal = np.asarray(normal, float)
    if normal.size == 2:
        return [np.array([-normal[1], normal[0]])]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(normal, ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return [t1, t2]


def build_footprints(geom, width_m, shape="flat", height_m=None, max_pts=9):
    """Finite-aperture element footprints on the finite-difference grid.

    Each element is a flat aperture lying in the plane tangent to the array
    surface, defined by its 3D shape:

    - ``"point"``  — single grid point (also the result of any width <= one cell).
    - ``"rect"``   — rectangle ``width_m`` (lateral, in-plane) by ``height_m``
      (elevation, along the array axis in 3D). In 2D this is a line segment of
      ``width_m``; height is the out-of-plane dimension and is ignored.
    - ``"disc"``   — elliptical disc with axes ``width_m`` x ``height_m``
      (a circular disc when they are equal). In 2D, a line segment.
    - ``"flat"``   — backwards-compatible alias for a square: rect with
      ``height_m = width_m``.

    For cylinder arrays the tangent frame is taken about the radial direction in
    the ring plane, so width runs circumferentially and height runs axially.
    The aperture is sampled onto grid points; on transmit every point is driven,
    on receive the pressure is averaged over them.

    Returns a list over elements of ``(idx_list, weights)`` where ``idx_list``
    holds grid-index tuples and ``weights`` sums to one.
    """
    pos = np.asarray(geom.element_positions, float)     # centred coords, metres
    dim = pos.shape[1]
    half = geom.domain_m / 2.0
    h = geom.h
    n = geom.n
    if height_m is None or shape == "flat":
        height_m = width_m
    point = shape == "point" or (width_m <= h and height_m <= h)
    is_cyl = getattr(geom, "array_type", "") == "cylinder"

    def _offsets(extent):
        npt = int(np.clip(round(extent / h) + 1, 1, max_pts))
        return np.linspace(-extent / 2.0, extent / 2.0, npt) if npt > 1 else np.zeros(1)

    footprints = []
    for k in range(len(pos)):
        if point:
            footprints.append(([geom.element_index(k)], np.array([1.0])))
            continue
        p = pos[k]
        pn = p.copy()
        if is_cyl:
            pn[2] = 0.0            # radial normal in the ring plane -> t2 axial
        norm = np.linalg.norm(pn)
        normal = pn / norm if norm > 0 else np.eye(dim)[0]
        tangents = _tangent_basis(normal)
        extents = [width_m] + ([height_m] if len(tangents) > 1 else [])
        grids = [_offsets(e) for e in extents]
        seen = {}
        for combo in np.array(np.meshgrid(*grids)).reshape(len(grids), -1).T:
            if shape == "disc":
                u = combo[0] / (width_m / 2.0 + 1e-30)
                v = (combo[1] / (height_m / 2.0 + 1e-30)) if len(combo) > 1 else 0.0
                if u * u + v * v > 1.0 + 1e-9:
                    continue
                # ellipse test in the tangent plane
            disp = sum(c * t for c, t in zip(combo, tangents))
            xyz = p + half + disp                       # absolute coords, metres
            ijk = tuple(int(np.clip(round(xyz[d] / h), 0, n - 1))
                        for d in range(dim - 1, -1, -1))  # axis order (z,y,x)
            seen[ijk] = True
        idx_list = list(seen.keys())
        footprints.append((idx_list, np.full(len(idx_list), 1.0 / len(idx_list))))
    return footprints


@dataclass
class GridArray:
    """Minimal array geometry defined directly by grid index tuples.

    Used when reconstructing a computational grid from a portable Dataset (which
    stores only physical element positions), so FWI can run on data that came
    from any source, including hardware.
    """

    idx: list
    n: int
    h: float
    radius_m: float
    domain_m: float

    @property
    def n_elements(self):
        return len(self.idx)

    def element_index(self, k):
        return self.idx[k]
