"""Robustness study for full-waveform COF inversion.

Repeats the grain-orientation inversion of run_cof_inversion.py under
degradations that a real experiment brings:

- additive noise on the observed traces (per-trace SNR 30 / 20 / 10 dB);
- fewer transmits (4 -> 2);
- observed data generated on a FINER grid than the inversion uses (breaks the
  inverse crime: different numerical dispersion and boundary discretisation,
  the same physical microstructure);
- the combination of fine-grid truth and 20 dB noise.

Each configuration runs the same optimizer (one coarse hemisphere sweep per
grain, then 15/7/3-degree local refinement) and reports the mean and max
c-axis error over the grains.

Run:  python examples/run_cof_robustness.py           (from simulation/)
Output: figures/cof_robustness.png + printed table.
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
N_GRAINS = 4
RADIUS = 0.010
DOM = 2 * RADIUS + 0.008
T_WIN = 22e-6
F0 = 0.3e6
N_INV = 30            # inversion grid
N_TRUE = 36           # fine grid for inverse-crime-free observed data
N_SWEEPS = 2          # coarse Gauss-Seidel sweeps (one is order-sensitive)


def make_setup(n_grid):
    h = DOM / (n_grid - 1)
    ring = CylinderArray(n_rings=3, per_ring=8, radius_m=RADIUS, height_m=0.012,
                         domain_m=DOM, h=h)
    dt = 0.4 * h / (an.ti_max_speed(MAT) * 1.02 * np.sqrt(3))
    nt = int(T_WIN / dt)
    clk = 100e6
    hp = max(1, round(clk / (2 * F0)))
    wav = td.transmit_chain(td.pulser_excitation(hp, 4, 2), clk, 1 / dt, F0, 0.6,
                            n_out=nt, tx_cut_hz=2.5 * F0)
    # Same physical microstructure at any grid: seeds and axes are drawn in
    # physical coordinates from the same rng seed.
    labels, axes_true, _ = phantom.voronoi_polycrystal_3d(
        (ring.n, ring.n, ring.n), N_GRAINS, 0.008, h,
        rng=np.random.default_rng(SEED), relax=1)
    return dict(ring=ring, n=ring.n, h=h, dt=dt, nt=nt, wav=wav,
                labels=labels, axes_true=axes_true)


def fmc(S, axes, src_list):
    Cm, rho = an.polycrystal_stiffness_3d(S["labels"], axes, material=MAT)
    ring, nt = S["ring"], S["nt"]
    data = np.zeros((len(src_list), nt, ring.n_elements))
    for i, s in enumerate(src_list):
        rec, _ = elastic3d.forward(Cm, rho, S["h"], S["dt"], nt,
                                   ring.element_index(s), S["wav"], ring.idx,
                                   source="explosive", record="pressure")
        data[i] = rec
    return data


def resample_time(data, dt_from, dt_to, nt_to):
    """Linear time-resampling of (n_tx, nt, n_rx) data onto a new grid."""
    n_tx, nt_from, n_rx = data.shape
    t_from = np.arange(nt_from) * dt_from
    t_to = np.arange(nt_to) * dt_to
    out = np.zeros((n_tx, nt_to, n_rx))
    for i in range(n_tx):
        for j in range(n_rx):
            out[i, :, j] = np.interp(t_to, t_from, data[i, :, j])
    return out


def add_noise(data, snr_db, rng):
    """Additive white noise at the given per-trace SNR (dB, RMS-based)."""
    rms = np.sqrt(np.mean(data ** 2, axis=1, keepdims=True))
    sigma = rms / (10.0 ** (snr_db / 20.0))
    return data + sigma * rng.standard_normal(data.shape)


def _hemi(a):
    return a if a[2] >= 0 else -a


def _neighbours(a, deg, n_az=6):
    a = np.asarray(a, float)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(a, ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(a, t1)
    r = np.radians(deg)
    return [_hemi((np.cos(r) * a + np.sin(r) * (np.cos(b) * t1 + np.sin(b) * t2))
                  / np.linalg.norm(np.cos(r) * a + np.sin(r) * (np.cos(b) * t1 + np.sin(b) * t2)))
            for b in np.linspace(0, 2 * np.pi, n_az, endpoint=False)]


def axis_error_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def invert(S, dobs, src_list, tag, objective="waveform"):
    """Per-grain Gauss-Seidel + refinement; returns recovered axes.

    ``objective``: "waveform" = per-trace normalised L2 on the waveforms;
    "envelope" = the same on Hilbert envelopes (phase-insensitive, tolerant of
    the numerical-dispersion mismatch between different discretisations).
    """
    from scipy.signal import hilbert

    def _obs(d):
        return np.abs(hilbert(d, axis=1)) if objective == "envelope" else d

    dref = _obs(dobs)
    tr_norm = np.sum(dref ** 2, axis=1) + 1e-30
    w = np.ones_like(tr_norm)
    for i, s in enumerate(src_list):
        w[i, s] = 0.0
    n_eval = [0]
    t0 = time.time()

    def J(axes):
        n_eval[0] += 1
        d = _obs(fmc(S, axes, src_list))
        return float(np.sum(w * np.sum((d - dref) ** 2, axis=1) / tr_norm))

    axes = np.tile([0.0, 0.0, 1.0], (N_GRAINS, 1))
    j_cur = J(axes)
    coarse = [_hemi(d) for d in _fibonacci_directions(16, hemisphere=True)]
    for _sweep in range(N_SWEEPS):
        for g in range(N_GRAINS):
            best_j, best_a = j_cur, axes[g].copy()
            for d in coarse:
                axes[g] = d
                jt = J(axes)
                if jt < best_j:
                    best_j, best_a = jt, np.array(d)
            axes[g] = best_a
            j_cur = best_j
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
    errs = [axis_error_deg(axes[g], S["axes_true"][g]) for g in range(N_GRAINS)]
    print(f"[{tag}] J_final={j_cur:.4f} evals={n_eval[0]} "
          f"({time.time() - t0:.0f}s) errors={np.round(errs, 1)} "
          f"mean={np.mean(errs):.1f} max={np.max(errs):.1f} deg", flush=True)
    return np.mean(errs), np.max(errs)


def main():
    S = make_setup(N_INV)
    src4 = list(range(0, S["ring"].n_elements, 6))          # 4 transmits
    src2 = src4[:2]
    print(f"inversion grid {S['n']}^3, {len(src4)} tx, nt={S['nt']}", flush=True)

    d_same = fmc(S, S["axes_true"], src4)                   # same-grid truth
    S_true = make_setup(N_TRUE)
    print(f"fine-truth grid {S_true['n']}^3, nt={S_true['nt']}", flush=True)
    d_fine_raw = fmc(S_true, S_true["axes_true"], src4)
    d_fine = resample_time(d_fine_raw, S_true["dt"], S["dt"], S["nt"])

    rng = np.random.default_rng(99)
    results = {}
    results["clean (control)"] = invert(S, d_same, src4, "clean")
    results["SNR 10 dB"] = invert(S, add_noise(d_same, 10, rng), src4, "snr10")
    results["2 transmits, clean"] = invert(S, d_same[:2], src2, "2tx")
    results["fine-grid (waveform)"] = invert(S, d_fine, src4, "fine-wf")
    results["fine-grid (envelope)"] = invert(S, d_fine, src4, "fine-env",
                                             objective="envelope")
    results["fine + 20 dB (envelope)"] = invert(S, add_noise(d_fine, 20, rng),
                                                src4, "fine+20dB-env",
                                                objective="envelope")

    print("\n=== COF robustness study (mean / max axis error, deg) ===")
    for k, (m, x) in results.items():
        print(f"  {k:22s}: {m:5.1f} / {x:5.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    means = [results[k][0] for k in names]
    maxs = [results[k][1] for k in names]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, means, 0.36, label="mean axis error")
    ax.bar(xpos + 0.18, maxs, 0.36, label="max axis error", alpha=0.7)
    ax.axhline(5, color="r", ls="--", lw=0.8, label="5 deg")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("c-axis error (deg)")
    ax.set_title("Full-waveform COF inversion: robustness")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "figures", "cof_robustness.png")
    fig.savefig(out, dpi=110)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
