"""3D isosurface render of a reconstruction: true model vs FWI, side by side.

Runs a small 3D ring-matrix FWI, then renders both the true sound-speed volume
and the recovered one as isosurfaces (translucent specimen shell + solid flaw).

Run:  python examples/run_render3d.py
Output: figures/render3d.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ringfwi import fwi, phantom, render3d
from ringfwi.geometry import CylinderArray
from ringfwi.sources import ricker


def main():
    h = 1.25e-3
    cyl = CylinderArray(n_rings=3, per_ring=12, radius_m=0.015, height_m=0.014,
                        domain_m=0.036, h=h)
    n = cyl.n
    dt = 1.2e-7
    nt = 300
    wavelet = ricker(nt, dt, 0.3e6)

    c_spec, c_coup, c_flaw = 3000.0, 1480.0, 2600.0
    c_bg = phantom.cylinder_background((n, n, n), c_spec, c_coup, 0.012, h)
    c_true = phantom.add_sphere(c_bg, (0.56, 0.5, 0.5), 0.005, c_flaw, h)

    src = list(range(0, cyl.n_elements, 3))
    dobs = fwi.forward_fmc(phantom.velocity_to_m(c_true), cyl, wavelet, dt, h, nt, src_list=src)

    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    mask = (np.hypot(xx - cc, yy - cc) <= 0.012 * 0.95).astype(float)
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2400.0))
    m_rec, hist = fwi.invert(phantom.velocity_to_m(c_bg), cyl, wavelet, dt, h, nt, dobs,
                             src_list=src, n_iter=8, step_frac=0.04,
                             update_mask=mask, m_bounds=m_bounds)
    c_rec = phantom.m_to_velocity(m_rec)
    print(f"FWI misfit {hist[0]:.3e} -> {hist[-1]:.3e} ({hist[-1]/hist[0]*100:.1f}%)")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    render3d.add_isosurfaces(ax1, render3d.surfaces(c_true, h, c_coup, c_spec, c_flaw),
                             c_true.shape, h, "True model")
    ax2 = fig.add_subplot(122, projection="3d")
    render3d.add_isosurfaces(ax2, render3d.surfaces(c_rec, h, c_coup, c_spec, c_flaw),
                             c_rec.shape, h, "FWI reconstruction")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "render3d.png")
    fig.savefig(out, dpi=120)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
