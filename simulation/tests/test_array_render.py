"""Smoke checks for the 3D array/transducer render."""

from __future__ import annotations

from ringfwi.geometry import CylinderArray, RingArray
from ringfwi.render3d import array3d_figure


def test_ring_shapes():
    ring = RingArray(n_elements=12, radius_m=0.024, domain_m=0.06, h=1e-3)
    fp = array3d_figure(ring, "point", 0.0)
    assert sum(t.type == "scatter3d" for t in fp.data) >= 2   # circle + markers
    fr = array3d_figure(ring, "rect", 0.002, 0.004)
    assert sum(t.type == "mesh3d" for t in fr.data) == 12     # one patch per element
    fd = array3d_figure(ring, "disc", 0.002, 0.002)
    assert sum(t.type == "mesh3d" for t in fd.data) == 12


def test_cylinder_rings():
    cyl = CylinderArray(n_rings=3, per_ring=8, radius_m=0.010, height_m=0.012,
                        domain_m=0.028, h=0.001)
    f = array3d_figure(cyl, "rect", 0.002, 0.003)
    circles = sum(t.type == "scatter3d" for t in f.data)
    patches = sum(t.type == "mesh3d" for t in f.data)
    assert circles == 3                                       # one circle per ring
    assert patches == 24


def test_polycrystal_render():
    import numpy as np
    from ringfwi import phantom
    from ringfwi.render3d import polycrystal_figure

    labels, axes, colat = phantom.voronoi_polycrystal_3d(
        (30, 30, 30), 8, 0.007, 0.0008, rng=np.random.default_rng(1))
    gv = np.degrees(np.arccos(np.clip(axes[:, 2], -1, 1)))
    melt = np.zeros((30, 30, 30), bool); melt[14:17, 14:17, 14:17] = True
    fig = polycrystal_figure(labels, gv, 0.0008, melt_mask=melt)
    n_mesh = sum(t.type == "mesh3d" for t in fig.data)
    present = len(np.unique(labels[labels >= 0]))
    assert n_mesh == present + 1                              # grains + melt pocket


if __name__ == "__main__":
    test_ring_shapes()
    test_cylinder_rings()
    print("array render checks passed")
