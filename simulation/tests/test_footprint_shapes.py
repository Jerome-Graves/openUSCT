"""Verify 3D transducer element shapes (rect / disc footprints)."""

from __future__ import annotations

import numpy as np

from ringfwi.geometry import CylinderArray, RingArray, build_footprints


def test_shapes_2d():
    ring = RingArray(n_elements=8, radius_m=0.024, domain_m=0.06, h=1.0e-3)
    pt = build_footprints(ring, 0.0, "point")
    assert all(len(i) == 1 for i, _ in pt)
    rect = build_footprints(ring, 0.003, "rect", height_m=0.001)
    assert max(len(i) for i, _ in rect) > 1          # a line of points in 2D
    flat = build_footprints(ring, 0.003, "flat")     # back-compat alias
    assert [len(i) for i, _ in flat] == [len(i) for i, _ in rect]
    for idxs, w in rect:
        assert np.isclose(w.sum(), 1.0)


def test_shapes_3d_rect_vs_disc():
    d = 0.028
    cyl = CylinderArray(n_rings=2, per_ring=8, radius_m=0.010, height_m=0.012,
                        domain_m=d, h=d / 33.0)
    rect = build_footprints(cyl, 0.004, "rect", height_m=0.004)
    disc = build_footprints(cyl, 0.004, "disc", height_m=0.004)
    tall = build_footprints(cyl, 0.002, "rect", height_m=0.006)
    n_rect = max(len(i) for i, _ in rect)
    n_disc = max(len(i) for i, _ in disc)
    n_tall = max(len(i) for i, _ in tall)
    print(f"rect {n_rect} pts, disc {n_disc} pts, tall-rect {n_tall} pts")
    assert n_rect > 1 and n_tall > 1
    assert n_disc <= n_rect                          # disc trims the corners
    # Tall rectangle must extend along the cylinder axis (z varies).
    zs = {ij[0] for ij, in_ in [(i, None) for fp in [tall[0][0]] for i in fp]}
    assert len(zs) > 1


if __name__ == "__main__":
    test_shapes_2d()
    test_shapes_3d_rect_vs_disc()
    print("footprint shape checks passed")
