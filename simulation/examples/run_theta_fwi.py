"""Voxel-wise orientation-field FWI of an ice polycrystal (2D).

The physically correct parameterisation for a known transversely isotropic
material: each voxel carries ONE angle, which encodes the full
direction-dependent velocity function through the rotated stiffness tensor.
The inversion recovers the c-axis angle map directly, with no grain-geometry
prior and no scalar-velocity approximation; the gradient is the exact elastic
adjoint chain-ruled through the analytic tensor-rotation derivative.

Like all gradient-based FWI this is a local method: the demo starts 15 deg
off the truth everywhere (in practice the grain-parameterised COF search of
ringfwi.cof provides the initialisation).

Run:  python examples/run_theta_fwi.py            (from simulation/)
Output: figures/theta_fwi.png
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ringfwi import anisotropy as an
from ringfwi import phantom
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker

BASE = an.ice_stiffness_2d(0.0)


def ang_err_deg(a, b):
    return np.degrees(0.5 * np.arccos(np.clip(np.cos(2 * (a - b)), -1, 1)))


def main():
    n = 56
    h = 4.0e-4
    nt = 340
    dt = 4.0e-8
    wav = ricker(nt, dt, 0.4e6)
    rho = np.full((n, n), an.ICE_RHO)
    ring = RingArray(n_elements=12, radius_m=0.0095, domain_m=(n - 1) * h, h=h)

    labels, angles, theta_true = phantom.voronoi_polycrystal(
        (n, n), 5, 0.02, h, rng=np.random.default_rng(5), relax=1)
    theta_true = np.where(labels >= 0, theta_true, angles[0])
    sources = [ring.element_index(s) for s in range(0, 12, 2)]

    Ct, _ = an.theta_stiffness_maps(theta_true, BASE)
    t0 = time.time()
    dobs = [an._grad_forward(Ct["C11"], Ct["C12"], Ct["C22"], Ct["C66"],
                             rho, h, dt, nt, s, wav, ring.idx,
                             Ct["C16"], Ct["C26"])[0]
            for s in sources]
    print(f"observed data in {time.time()-t0:.0f}s", flush=True)

    yy, xx = np.mgrid[0:n, 0:n].astype(float) * h
    cc = (n - 1) * h / 2
    inside = np.hypot(xx - cc, yy - cc) <= 0.0080

    theta0 = theta_true + np.radians(15.0)
    err0 = float(np.mean(ang_err_deg(theta0, theta_true)[inside]))
    theta_rec, hist = an.invert_theta(
        theta0, BASE, rho, h, dt, nt, sources, wav, ring.idx, dobs,
        n_iter=40, step_rad=0.2, smooth_sigma=1.0,
        update_mask=inside.astype(float), verbose=True)
    err1 = float(np.mean(ang_err_deg(theta_rec, theta_true)[inside]))
    print(f"misfit {hist[0]:.3e} -> {hist[-1]:.3e} "
          f"({hist[-1]/hist[0]*100:.1f}%)")
    print(f"interior mean angle error: {err0:.1f} -> {err1:.1f} deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
    masked = lambda a: np.where(inside, np.degrees(a) % 180.0, np.nan)  # noqa: E731
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.0))
    for a_, m_, ttl in ((ax[0], masked(theta_true), "True c-axis angle (deg)"),
                        (ax[1], masked(theta0), "Start (+15 deg)"),
                        (ax[2], masked(theta_rec), "Recovered")):
        im = a_.imshow(m_, origin="lower", extent=ext, cmap="twilight",
                       vmin=0, vmax=180)
        a_.set_xlabel("mm")
        a_.set_title(ttl)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    e = np.where(inside, ang_err_deg(theta_rec, theta_true), np.nan)
    im = ax[3].imshow(e, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=20)
    ax[3].set_title(f"Angle error (mean {err1:.1f} deg)")
    ax[3].set_xlabel("mm")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "figures", "theta_fwi.png")
    fig.savefig(out, dpi=110)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
