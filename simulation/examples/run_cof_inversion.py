"""Full-waveform crystal-orientation-fabric (COF) inversion.

Recovers the 3D c-axis of every grain in an ice polycrystal from elastic
full-matrix-capture data, using the full 3D anisotropic elastic solver as the
forward model. The grain geometry (Voronoi labels) and the single-crystal
material are treated as known — the experimental analogue is grain boundaries
mapped optically from a thin section, orientations then determined by
ultrasound. The unknowns are just two angles per grain, so no adjoint is
needed: a per-grain Gauss-Seidel search over a hemisphere of candidate axes,
followed by two local refinement levels.

Objective: per-trace normalised L2 misfit (each source-receiver path weighted
equally; the transmitter's own trace excluded).

Run:  python examples/run_cof_inversion.py            (from simulation/)
Output: figures/cof_inversion.png + printed angle-error table.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ringfwi import anisotropy as an
from ringfwi import elastic3d, phantom
from ringfwi import transducer as td
from ringfwi.geometry import CylinderArray, _fibonacci_directions

MAT = an.ICE_MATERIAL
SEED = 7
N_GRAINS = 6
N_GRID = 26
F0 = 0.3e6


def make_setup():
    radius = 0.010
    dom = 2 * radius + 0.008
    h = dom / (N_GRID - 1)
    ring = CylinderArray(n_rings=3, per_ring=8, radius_m=radius, height_m=0.012,
                         domain_m=dom, h=h)
    n = ring.n
    dt = 0.4 * h / (an.ti_max_speed(MAT) * 1.02 * np.sqrt(3))
    nt = int(22e-6 / dt)
    clk = 100e6
    hp = max(1, round(clk / (2 * F0)))
    wav = td.transmit_chain(td.pulser_excitation(hp, 4, 2), clk, 1 / dt, F0, 0.6,
                            n_out=nt, tx_cut_hz=2.5 * F0)
    labels, axes_true, _ = phantom.voronoi_polycrystal_3d(
        (n, n, n), N_GRAINS, 0.008, h, rng=np.random.default_rng(SEED), relax=1)
    src_list = list(range(0, ring.n_elements, 4))          # 6 transmits
    return ring, n, h, dt, nt, wav, labels, axes_true, src_list


def fmc(labels, axes, ring, h, dt, nt, wav, src_list):
    Cm, rho = an.polycrystal_stiffness_3d(labels, axes, material=MAT)
    data = np.zeros((len(src_list), nt, ring.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = elastic3d.forward(Cm, rho, h, dt, nt, ring.element_index(s),
                                   wav, ring.idx, source="explosive",
                                   record="pressure")
        data[i] = rec
    return data


def _hemi(a):
    return a if a[2] >= 0 else -a


def _neighbours(a, deg, n_az=6):
    """Candidate axes tilted ``deg`` from ``a`` at ``n_az`` azimuths."""
    a = np.asarray(a, float)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(a, ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(a, t1)
    r = np.radians(deg)
    out = []
    for b in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
        d = np.cos(r) * a + np.sin(r) * (np.cos(b) * t1 + np.sin(b) * t2)
        out.append(_hemi(d / np.linalg.norm(d)))
    return out


def axis_error_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def main():
    ring, n, h, dt, nt, wav, labels, axes_true, src_list = make_setup()
    print(f"setup: {N_GRAINS} grains, grid {n}^3, {len(src_list)} tx x "
          f"{ring.n_elements} rx, nt={nt}", flush=True)

    t0 = time.time()
    dobs = fmc(labels, axes_true, ring, h, dt, nt, wav, src_list)
    tr_norm = np.sum(dobs ** 2, axis=1) + 1e-30
    w = np.ones_like(tr_norm)
    for i, s in enumerate(src_list):
        w[i, s] = 0.0
    print(f"observed data acquired in {time.time() - t0:.0f}s", flush=True)

    n_eval = [0]

    def J(axes):
        n_eval[0] += 1
        d = fmc(labels, axes, ring, h, dt, nt, wav, src_list)
        return float(np.sum(w * np.sum((d - dobs) ** 2, axis=1) / tr_norm))

    # Start from all-vertical axes (no prior orientation knowledge).
    axes = np.tile([0.0, 0.0, 1.0], (N_GRAINS, 1))
    j_cur = J(axes)
    print(f"J(start, all vertical) = {j_cur:.4f}", flush=True)

    # Stage 1: per-grain global search over a coarse hemisphere.
    coarse = [_hemi(d) for d in _fibonacci_directions(16, hemisphere=True)]
    for sweep in range(2):
        for g in range(N_GRAINS):
            best_j, best_a = j_cur, axes[g].copy()
            for d in coarse:
                axes[g] = d
                jt = J(axes)
                if jt < best_j:
                    best_j, best_a = jt, np.array(d)
            axes[g] = best_a
            j_cur = best_j
            print(f"  sweep {sweep} grain {g}: J={j_cur:.4f} "
                  f"err={axis_error_deg(axes[g], axes_true[g]):.1f} deg "
                  f"[{n_eval[0]} evals, {time.time() - t0:.0f}s]", flush=True)

    # Stage 2: local refinement at decreasing tilt levels.
    for deg in (15.0, 7.0, 3.0):
        for g in range(N_GRAINS):
            improved = True
            while improved:
                improved = False
                for d in _neighbours(axes[g], deg):
                    old = axes[g].copy()
                    axes[g] = d
                    jt = J(axes)
                    if jt < j_cur:
                        j_cur = jt
                        improved = True
                    else:
                        axes[g] = old
            print(f"  refine {deg:.0f} deg grain {g}: J={j_cur:.4f} "
                  f"err={axis_error_deg(axes[g], axes_true[g]):.1f} deg "
                  f"[{n_eval[0]} evals, {time.time() - t0:.0f}s]", flush=True)

    errs = [axis_error_deg(axes[g], axes_true[g]) for g in range(N_GRAINS)]
    print("\n=== COF inversion result ===")
    print(f"total: {n_eval[0]} forward evaluations, {time.time() - t0:.0f}s, "
          f"final J = {j_cur:.4f}")
    for g in range(N_GRAINS):
        print(f"  grain {g}: true axis {np.round(axes_true[g], 2)}  "
              f"recovered {np.round(axes[g], 2)}  error {errs[g]:.1f} deg")
    print(f"  mean axis error: {np.mean(errs):.1f} deg | max {np.max(errs):.1f} deg")

    # Figure: colatitude maps (mid slice) + per-grain error bars.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def colat_map(ax_arr):
        colat = np.degrees(np.arccos(np.clip(np.abs(ax_arr[:, 2]), 0, 1)))
        vol = np.where(labels >= 0, colat[np.clip(labels, 0, None)], np.nan)
        return vol[n // 2]

    fig, axp = plt.subplots(1, 3, figsize=(13, 4.2))
    for a_, m_, ttl in ((axp[0], colat_map(axes_true), "True c-axis colatitude (deg)"),
                        (axp[1], colat_map(axes), "Recovered colatitude")):
        im = a_.imshow(m_, origin="lower", cmap="twilight", vmin=0, vmax=90)
        a_.set_title(ttl)
        fig.colorbar(im, ax=a_, fraction=0.046)
    axp[2].bar(range(N_GRAINS), errs, color="#4488cc")
    axp[2].set_title("Axis error per grain (deg)")
    axp[2].set_xlabel("grain"); axp[2].axhline(5, color="r", ls="--", lw=0.8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "figures", "cof_inversion.png")
    fig.savefig(out, dpi=110)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
