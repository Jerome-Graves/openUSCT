"""OpenUSCT Studio: a pure-Python control and analysis GUI.

A MATLAB-free equivalent of the UARP workflow: configure the cylindrical array,
transducer elements, excitation chain, and sample in tabs; run a simulated 3D
full-matrix capture; image it (total focusing method); reconstruct it (full
waveform inversion); and export it in the UARP/UDSP format. Runs on the
``ringfwi`` stack and uses the C++ backend automatically when it is available.

Samples are 3D: a uniform specimen (with optional flaw), or a 3D Voronoi
polycrystal of anisotropic grains (ice Ih and other TI materials, preset or
custom constants) — the crystal-orientation-fabric
scenario, acquired with the full 3D anisotropic elastic solver (21-component
stiffness per grain). The transmit is FPGA-first: a discrete logic-level drive
shaped by filters and the transducer response, and every acquisition streams
through the rx_capture RTL in Icarus Verilog.

Run:  streamlit run software/gui/app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "simulation"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from ringfwi import anisotropy, elastic3d, fwi, imaging, phantom, render3d
from ringfwi import transducer as td
from ringfwi.dataset import ArrayGeometry, Dataset
from ringfwi.geometry import CylinderArray, build_footprints
from ringfwi.uarp_format import to_uarp_set

# Self-heal a stale server: Streamlit re-reads this script on every rerun but
# keeps imported modules cached, so a server started before a library change
# serves old modules. If a required symbol is missing, purge and re-import.
if not (hasattr(anisotropy, "TI_MATERIALS") and hasattr(td, "transmit_chain")
        and hasattr(render3d, "polycrystal_figure")):
    import importlib
    for _name in [m for m in list(sys.modules)
                  if m == "ringfwi" or m.startswith("ringfwi.")]:
        del sys.modules[_name]
    from ringfwi import anisotropy, elastic3d, fwi, imaging, phantom, render3d
    from ringfwi import transducer as td
    from ringfwi.dataset import ArrayGeometry, Dataset
    from ringfwi.geometry import CylinderArray, build_footprints
    from ringfwi.uarp_format import to_uarp_set

import hw_cosim

try:
    import uap as _backend
    BACKEND, BACKEND_NAME = _backend, "C++ (libuap)"
except Exception:
    BACKEND, BACKEND_NAME = fwi, "Python"

st.set_page_config(page_title="OpenUSCT Studio", layout="wide")
st.title("OpenUSCT Studio")
st.caption(f"3D cylindrical-array ultrasound acquisition and reconstruction, pure Python. Backend: {BACKEND_NAME}.")

tab_arr, tab_exc, tab_smp, tab_acq, tab_img, tab_fwi = st.tabs(
    ["Array & Transducer", "Excitation & Filters", "Sample", "Acquisition",
     "Imaging (TFM)", "Reconstruction (FWI)"])


# ---- Configuration widgets (first pass) ------------------------------------
with tab_arr:
    ca1, ca2 = st.columns(2)
    with ca1:
        st.subheader("Array (3D cylinder)")
        per_ring = st.slider("Elements per ring", 6, 16, 8, 1)
        n_rings = st.slider("Rings", 1, 4, 3, 1)
        radius_mm = st.slider("Ring radius (mm)", 6, 16, 10)
        if n_rings > 1:
            height_mm = st.slider("Axial span (mm)", 6, 18, 12)
        else:
            height_mm = 0
            st.caption("A single ring sits on the mid-plane; axial span does not apply.")
    with ca2:
        st.subheader("Grid & sampling")
        n_grid = st.slider("Grid points per axis", 28, 60, 34, 2,
                           help="Finer grid = more accurate but slower (cubic cost in 3D).")
        nt = st.slider("Samples (nt)", 150, 900, 220, 10)

    ct1, ct2 = st.columns(2)
    with ct1:
        st.subheader("Element geometry")
        el_shape = st.radio("Element shape", ["Point", "Rectangular", "Disc"], horizontal=True,
                            help="Rectangular and Disc are finite apertures sampled onto the "
                                 "grid and simulated on transmit and receive.")
    with ct2:
        st.subheader("Element size")
        if el_shape == "Disc":
            el_w_mm = el_h_mm = st.slider("Element diameter (mm)", 0.5, 6.0, 2.0, 0.5,
                                          help="Diameter of the circular element face.")
        elif el_shape == "Rectangular":
            el_w_mm = st.slider("Element width (mm)", 0.5, 6.0, 2.0, 0.5,
                                help="Lateral size, circumferential on the cylinder.")
            el_h_mm = st.slider("Element height (mm)", 0.5, 6.0, 2.0, 0.5,
                                help="Elevation size, along the cylinder axis.")
        else:
            el_w_mm = el_h_mm = 0.0
            st.caption("Point elements have no size.")
    arr_preview = st.container()

with tab_exc:
    st.caption("The FPGA pulser cannot produce an ideal analytic wavelet: it drives "
               "discrete levels {-1, 0, +1} at the logic clock. The acoustic output is "
               "that discrete signal shaped by the TX filter and the transducer response.")
    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        st.markdown("**Input signal (FPGA drive)**")
        sig_type = st.radio("Signal type",
                            ["Bipolar square burst (HV pulser)", "Unipolar spike",
                             "Square chirp (coded excitation)"])
        f0_mhz = st.select_slider("Centre frequency (MHz)", [0.25, 0.3, 0.5, 1.0], value=0.3)
        clk_mhz = st.select_slider("FPGA clock (MHz)", [50, 100, 200], value=100)
        if sig_type.startswith("Bipolar"):
            n_cycles = st.slider("Burst cycles", 1, 6, 2, 1)
            dead_clk = st.slider("Dead time (clocks)", 0, 10, 2, 1,
                                 help="Shoot-through-safe gap between the pulser half-bridges.")
        elif sig_type.startswith("Square chirp"):
            chirp_lo = st.slider("Chirp start (x f0)", 0.3, 1.0, 0.5, 0.05)
            chirp_hi = st.slider("Chirp end (x f0)", 1.0, 2.5, 1.5, 0.05)
            chirp_cyc = st.slider("Chirp length (cycles of f0)", 2, 16, 6, 1)
    with ce2:
        st.markdown("**Transducer response**")
        frac_bw = st.slider("Fractional bandwidth", 0.3, 1.0, 0.6, 0.05,
                            help="-6 dB fractional bandwidth of the element around f0. "
                                 "Lower = longer ringing; higher = short, broadband.")
        st.markdown("**TX output filter**")
        tx_filt_on = st.checkbox("Apply TX low-pass", True,
                                 help="Output/matching filter after the HV pulser.")
        tx_cut_x = st.slider("TX cutoff (x f0)", 1.5, 4.0, 2.5, 0.25)
    with ce3:
        st.markdown("**RX front-end filter**")
        rx_filt_on = st.checkbox("Apply RX band-pass", True,
                                 help="Analogue front-end band-pass applied to every "
                                      "received channel before the ADC.")
        rx_lo_x = st.slider("RX low cut (x f0)", 0.1, 0.8, 0.3, 0.05)
        rx_hi_x = st.slider("RX high cut (x f0)", 1.2, 3.0, 2.2, 0.1)
    exc_preview = st.container()

with tab_smp:
    st.subheader("Sample definition")
    sample_type = st.radio("Sample type", ["Uniform specimen (optional flaw)",
                                           "Voronoi polycrystal (anisotropic grains)"])
    poly = sample_type.startswith("Voronoi")

    cs1, cs2, cs3 = st.columns(3)
    if poly:
        with cs1:
            st.markdown("**Specimen & microstructure**")
            spec_r_mm = st.slider("Specimen radius (mm)", 5, 14, 8,
                                  help="The polycrystal cylinder fills the testing zone.")
            c_coup = st.slider("Couplant speed (m/s)", 1400, 1600, 1480, 10)
            n_grains = st.slider("Grains", 4, 60, 14, 1)
            seed = st.number_input("Random seed", 0, 9999, 3, 1)
        with cs2:
            st.markdown("**Anisotropic material**")
            mat_choice = st.selectbox(
                "Material preset",
                list(anisotropy.TI_MATERIALS) + ["Custom (enter constants)"],
                help="Transversely isotropic (hexagonal) single-crystal constants; "
                     "the preset values are typical literature figures.")
            if mat_choice.startswith("Custom"):
                mc1, mc2 = st.columns(2)
                C11g = mc1.number_input("C11 (GPa)", 0.1, 2000.0, 13.93, 0.1)
                C33g = mc2.number_input("C33 (GPa)", 0.1, 2000.0, 15.01, 0.1)
                C44g = mc1.number_input("C44 (GPa)", 0.05, 1000.0, 3.01, 0.1)
                C12g = mc2.number_input("C12 (GPa)", -500.0, 1500.0, 7.08, 0.1)
                C13g = mc1.number_input("C13 (GPa)", -500.0, 1500.0, 5.77, 0.1)
                rho_in = mc2.number_input("Density (kg/m3)", 100.0, 25000.0, 917.0, 1.0)
                material = dict(C11=C11g * 1e9, C33=C33g * 1e9, C44=C44g * 1e9,
                                C12=C12g * 1e9, C13=C13g * 1e9, rho=rho_in)
                mat_label = "custom TI material"
            else:
                material = anisotropy.TI_MATERIALS[mat_choice]
                mat_label = mat_choice
                st.caption(f"C11 {material['C11']/1e9:.1f} | C33 {material['C33']/1e9:.1f} | "
                           f"C44 {material['C44']/1e9:.1f} | C12 {material['C12']/1e9:.1f} | "
                           f"C13 {material['C13']/1e9:.1f} GPa | "
                           f"rho {material['rho']:.0f} kg/m3")
            if np.linalg.eigvalsh(anisotropy.ti_stiffness_6(**material)).min() <= 0:
                st.error("These constants are not physically admissible (the stiffness "
                         "tensor must be positive definite). Adjust C11/C12/C13/C33/C44.")
                st.stop()
        with cs3:
            st.markdown("**Isotropic pocket (region)**")
            melt_on = st.checkbox("Include fluid pocket (e.g. melt inclusion)", False)
            melt_fx = st.slider("Pocket x (fraction)", 0.30, 0.70, 0.58, 0.02)
            melt_fy = st.slider("Pocket y (fraction)", 0.30, 0.70, 0.46, 0.02)
            melt_r_mm = st.slider("Pocket radius (mm)", 1, 8, 3)
        flaw_on = False
        c_spec, flaw_c = 3970, 1480          # placeholders for shared code paths
    else:
        with cs1:
            st.markdown("**Specimen**")
            c_spec = st.slider("Specimen speed (m/s)", 2000, 4000, 3000, 50)
            c_coup = st.slider("Couplant speed (m/s)", 1400, 1600, 1480, 10)
            spec_r_mm = st.slider("Specimen radius (mm)", 5, 14, 8)
        with cs2:
            st.markdown("**Flaw region**")
            flaw_on = st.checkbox("Include flaw", True)
            flaw_c = st.slider("Flaw speed (m/s)", 2000, 3500, 2650, 50)
            flaw_r_mm = st.slider("Flaw radius (mm)", 2, 8, 4)
        with cs3:
            st.markdown("**Flaw position**")
            flaw_fx = st.slider("Flaw x (fraction)", 0.30, 0.70, 0.58, 0.02)
            flaw_fy = st.slider("Flaw y (fraction)", 0.30, 0.70, 0.50, 0.02)
        melt_on = False
    smp_preview = st.container()


# ---- Build geometry and models ---------------------------------------------
radius_m = radius_mm / 1000.0
spec_r_m = spec_r_mm / 1000.0
f0 = f0_mhz * 1e6
cmax = (anisotropy.ti_max_speed(material) * 1.02 if poly
        else max(c_spec, c_coup, flaw_c if flaw_on else 0))

domain_m = 2 * radius_m + 0.008
h = domain_m / (n_grid - 1)
ring = CylinderArray(n_rings=n_rings, per_ring=per_ring, radius_m=radius_m,
                     height_m=height_mm / 1000.0, domain_m=domain_m, h=h)
n = ring.n
dt = (0.4 if poly else 0.5) * h / (cmax * np.sqrt(3))
n_elem = ring.n_elements
src_list = list(range(0, n_elem, 2))          # subset of transmits to stay quick
ndim = 3

melt_mask = None
if poly:
    labels, axes3, theta_map = phantom.voronoi_polycrystal_3d(
        (n, n, n), n_grains, spec_r_m, h, rng=np.random.default_rng(int(seed)), relax=1)
    if melt_on:
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
        D = (n - 1) * h
        melt_mask = np.sqrt((xx - melt_fx * D) ** 2 + (yy - melt_fy * D) ** 2
                            + (zz - 0.5 * D) ** 2) <= melt_r_mm / 1000.0
    # Full 21-component stiffness for the 3D anisotropic elastic solver.
    Cmaps3, rho_map3 = anisotropy.polycrystal_stiffness_3d(
        labels, axes3, c_couplant=c_coup, fluid_mask=melt_mask, material=material)
    # Apparent per-grain qP map: display + acoustic-FWI reference only.
    c_true = anisotropy.polycrystal_apparent_speed_3d(
        labels, axes3, c_couplant=c_coup, fluid_mask=melt_mask, material=material)
    c_bg = phantom.cylinder_background(
        (n, n, n), float(np.median(c_true[labels >= 0])), c_coup, spec_r_m, h)
else:
    c_bg = phantom.cylinder_background((n, n, n), c_spec, c_coup, spec_r_m, h)
    c_true = phantom.add_sphere(c_bg, (flaw_fx, flaw_fy, 0.5), flaw_r_mm / 1000.0, flaw_c, h) if flaw_on else c_bg

shape_key = {"Point": "point", "Rectangular": "rect", "Disc": "disc"}[el_shape]
footprints = None if el_shape == "Point" else build_footprints(
    ring, el_w_mm / 1000.0, shape_key, height_m=el_h_mm / 1000.0)
elem_abs = np.asarray(ring.element_positions) + domain_m / 2.0   # absolute, metres

# --- FPGA transmit chain: discrete drive -> TX filter -> transducer ----------
clk_hz = clk_mhz * 1e6
fs = 1.0 / dt
half_period = max(1, int(round(clk_hz / (2.0 * f0))))
if sig_type.startswith("Bipolar"):
    exc_clk = td.pulser_excitation(half_period, 2 * n_cycles, dead_clk)
elif sig_type.startswith("Unipolar"):
    exc_clk = td.unipolar_pulse(half_period)
else:
    exc_clk = td.square_chirp(clk_hz, chirp_lo * f0, chirp_hi * f0, chirp_cyc / f0)
wavelet = td.transmit_chain(exc_clk, clk_hz, fs, f0, frac_bw, n_out=nt,
                            tx_cut_hz=(tx_cut_x * f0 if tx_filt_on else None))


def apply_rx_filter(arr, axis):
    """RX analogue front-end band-pass, applied per channel."""
    if not rx_filt_on:
        return arr
    return td.bandpass(arr, fs, rx_lo_x * f0, rx_hi_x * f0, order=2, axis=axis)


# Reconstruction must model what the receiver recorded: the effective wavelet
# seen through the RX filter (linear chain, so filtering the wavelet is
# equivalent to filtering the data).
wavelet_eff = apply_rx_filter(wavelet, axis=0)
# For ELASTIC data reconstructed with the acoustic engine, the effective source
# differs: stress injection radiates the negative time-derivative of the
# acoustic pressure-source response (verified numerically in water, corr -0.994
# at zero lag). Without this the inversion drifts systematically.
if poly:
    _wd = -np.gradient(wavelet, dt)
    _wd = _wd / (np.abs(_wd).max() + 1e-30)
    wavelet_rec = apply_rx_filter(_wd, axis=0)
else:
    wavelet_rec = wavelet_eff

ext = [0, (n - 1) * h * 1e3, 0, (n - 1) * h * 1e3]
if poly:
    ice = labels >= 0
    vlo, vhi = float(c_true[ice].min()) - 30, float(c_true[ice].max()) + 30
else:
    vlo, vhi = (min(flaw_c, c_spec) - 100 if flaw_on else c_coup), c_spec + 100


def run_acquisition(m, progress=None):
    """Full-matrix capture with the sample's physics, one transmit at a time.

    ``progress(done, total, elapsed_s)`` is called after each transmit so the
    GUI can show completion and an expected time remaining.
    """
    data = np.zeros((len(src_list), nt, n_elem))
    t0 = time.time()
    for i, s in enumerate(src_list):
        if poly:                          # full 3D anisotropic elastic
            src_pts = (list(zip(footprints[s][0], footprints[s][1]))
                       if footprints is not None else None)
            rec, _ = elastic3d.forward(Cmaps3, rho_map3, h, dt, nt,
                                       ring.element_index(s), wavelet, ring.idx,
                                       source="explosive", record="pressure",
                                       src_pts=src_pts, rec_groups=footprints)
            data[i] = rec
        elif footprints is not None:
            data[i] = fwi.forward_fmc(m, ring, wavelet, dt, h, nt, src_list=[s],
                                      footprints=footprints)[0]
        else:
            data[i] = BACKEND.forward_fmc(m, ring, wavelet, dt, h, nt, src_list=[s])[0]
        if progress is not None:
            progress(i + 1, len(src_list), time.time() - t0)
    return data


def _overlay_elements(ax, cols):
    """Scatter element positions (mm) onto a 2D axis using coordinate columns."""
    p = elem_abs * 1e3
    ax.scatter(p[:, cols[0]], p[:, cols[1]], s=14, facecolors="none",
               edgecolors="#00e5ff", linewidths=0.9, zorder=5)


def show_model(vol, title, cmap="viridis", vmin=None, vmax=None, elements=False):
    """Three orthogonal central slices of a 3D volume, optional element overlay."""
    nz, ny, nx = vol.shape
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.9))
    slices = [(vol[nz // 2], "xy", (0, 1)), (vol[:, ny // 2, :], "xz", (0, 2)),
              (vol[:, :, nx // 2], "yz", (1, 2))]
    for a, (sl, nm, cols) in zip(ax, slices):
        im = a.imshow(sl, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
        if elements:
            _overlay_elements(a, cols)
        a.set_title(f"{title} ({nm})"); a.set_xlabel("mm")
    fig.colorbar(im, ax=ax, fraction=0.02)
    return fig


def show_fig(fig, container=None):
    """Render a matplotlib figure at a fixed, modest pixel width (not stretched)."""
    w, _ = fig.get_size_inches()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    (container or st).image(buf, width=int(min(640, 105 * w)))


def plotly_hover_slice(fig, height=520, cap=None):
    """Interactive 3D chart with hover-driven horizontal slicing.

    Moving the cursor up/down over the chart cuts the geometry at the matching
    height; leaving the chart restores the full view. Meshes are clamped to the
    cut plane (watertight stays closed). ``cap`` optionally supplies a voxel
    label volume + colour table: the true cross-section layer is then drawn on
    the cut plane, so overlapping grains cannot fight — the top is the exact
    material section. Everything runs client-side, no server round-trips.

    cap = dict(vol=labels.tolist() (nz,ny,nx; -1 = empty), h_mm=grid step in mm,
               colors={str(label): "rgb(...)"}).
    """
    import json
    # components.v1.html is deprecated but remains the only Streamlit API that
    # executes inline JS; st.html sanitises scripts and st.iframe would need the
    # multi-megabyte mesh JSON in a data URL (browser-limited).
    import streamlit.components.v1 as components

    import base64

    fig.update_layout(template="plotly_dark", height=height - 20,
                      paper_bgcolor="rgba(0,0,0,0)")
    spec = json.loads(fig.to_json())

    def _plain(v):
        """Decode plotly>=6 typed-array encoding {dtype, bdata} to a list."""
        if isinstance(v, dict) and "bdata" in v:
            arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
            if "shape" in v:
                arr = arr.reshape([int(s) for s in str(v["shape"]).split(",")])
            return arr.tolist()
        return v

    for tr in spec["data"]:
        for key in ("x", "y", "z", "i", "j", "k", "intensity"):
            if key in tr:
                tr[key] = _plain(tr[key])
    # Freeze the scene axes so cutting geometry never rescales the view.
    spans = {}
    for ax in ("x", "y", "z"):
        vals = [v for tr in spec["data"] if tr.get("type") in ("mesh3d", "scatter3d")
                for v in tr.get(ax, [])]
        lo, hi = min(vals), max(vals)
        pad = 0.04 * (hi - lo + 1e-30)
        spans[ax] = (lo - pad, hi + pad)
    scene = spec["layout"].setdefault("scene", {})
    smax = max(s[1] - s[0] for s in spans.values())
    for ax in ("x", "y", "z"):
        scene.setdefault(f"{ax}axis", {})["range"] = list(spans[ax])
        scene[f"{ax}axis"]["autorange"] = False
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = {ax: (spans[ax][1] - spans[ax][0]) / smax
                            for ax in ("x", "y", "z")}
    # Placeholder trace for the voxel-accurate cut face (filled by the JS).
    if cap is not None:
        spec["data"].append(dict(type="mesh3d", x=[0.0], y=[0.0], z=[0.0],
                                 i=[], j=[], k=[], visible=False,
                                 flatshading=True, hoverinfo="skip",
                                 name="cut face", showlegend=False))
    html = """
<div id="wrap" style="width:100%;height:HEIGHTpx;"></div>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
const spec = SPEC;
const CAP = CAPSPEC;
const gd = document.getElementById('wrap');
Plotly.newPlot(gd, spec.data, spec.layout, {responsive: true});
// Slice by CLAMPING vertices just below the cut plane (watertight stays
// closed), then draw the TRUE voxel cross-section on the plane itself so
// overlapping collapsed grain tops can never show through. Scatter traces
// (the transducer elements) are point-filtered by the same cut.
const meshIdx = [], orig = [];
const scatIdx = [], scatOrig = [];
const capIdx = CAP ? spec.data.length - 1 : -1;
let zmin = Infinity, zmax = -Infinity;
spec.data.forEach((tr, ti) => {
  if (ti === capIdx) return;
  if (tr.type === 'mesh3d') {
    meshIdx.push(ti);
    const z = tr.z.slice();
    let lo = Infinity;
    for (const v of z) { if (v < lo) lo = v; if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
    orig.push({z: z, lo: lo});
  } else if (tr.type === 'scatter3d' && tr.mode && tr.mode.includes('markers')) {
    scatIdx.push(ti);
    scatOrig.push({x: tr.x.slice(), y: tr.y.slice(), z: tr.z.slice()});
  }
});
function buildCap(zcut) {
  const hmm = CAP.h_mm, vol = CAP.vol, nz = vol.length;
  const layer = Math.min(nz - 1, Math.max(0, Math.round(zcut / hmm)));
  const L = vol[layer], ny = L.length, nx = L[0].length;
  const X = [], Y = [], Z = [], I = [], J = [], K = [], FC = [];
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const lab = L[iy][ix];
      if (lab < 0 && !(String(lab) in CAP.colors)) continue;
      const col = CAP.colors[String(lab)];
      if (!col) continue;
      const v0 = X.length;
      const x0 = (ix - 0.5) * hmm, x1 = (ix + 0.5) * hmm;
      const y0 = (iy - 0.5) * hmm, y1 = (iy + 0.5) * hmm;
      X.push(x0, x1, x1, x0); Y.push(y0, y0, y1, y1); Z.push(zcut, zcut, zcut, zcut);
      I.push(v0, v0); J.push(v0 + 1, v0 + 2); K.push(v0 + 2, v0 + 3);
      FC.push(col, col);
    }
  }
  return {X: X, Y: Y, Z: Z, I: I, J: J, K: K, FC: FC};
}
let pending = false, frac = 1.0, armed = false;
function applyCut(fr) {
  const zcut = zmin + fr * (zmax - zmin);
  const eps = CAP ? 0.12 * CAP.h_mm : 0.0;
  const zclamp = zcut - eps;                   // tuck collapsed tops under the cap
  const Z = [], VIS = [];
  orig.forEach(o => {
    if (o.lo >= zclamp) { Z.push(o.z); VIS.push(false); }
    else if (fr >= 0.999) { Z.push(o.z); VIS.push(true); }
    else { Z.push(o.z.map(v => v > zclamp ? zclamp : v)); VIS.push(true); }
  });
  Plotly.restyle(gd, {z: Z, visible: VIS}, meshIdx);
  if (scatIdx.length) {                        // transducer elements follow the cut
    const SX = [], SY = [], SZ = [];
    scatOrig.forEach(o => {
      if (fr >= 0.999) { SX.push(o.x); SY.push(o.y); SZ.push(o.z); return; }
      const xx = [], yy = [], zz = [];
      for (let p = 0; p < o.z.length; p++) {
        if (o.z[p] <= zcut) { xx.push(o.x[p]); yy.push(o.y[p]); zz.push(o.z[p]); }
      }
      SX.push(xx); SY.push(yy); SZ.push(zz);
    });
    Plotly.restyle(gd, {x: SX, y: SY, z: SZ}, scatIdx);
  }
  if (CAP) {
    if (fr >= 0.999) {
      Plotly.restyle(gd, {visible: [false]}, [capIdx]);
    } else {
      const c = buildCap(zcut);
      Plotly.restyle(gd, {x: [c.X], y: [c.Y], z: [c.Z], i: [c.I], j: [c.J],
                          k: [c.K], facecolor: [c.FC], visible: [true]}, [capIdx]);
    }
  }
}
// Slicing arms only once the cursor actually touches the rendered geometry.
gd.on('plotly_hover', () => { armed = true; });
const CUT_LIFT = 0.10;  // keep the cut a little ABOVE the cursor, so the cut
                        // face stays under the pointer and grain hover works
gd.addEventListener('mousemove', ev => {
  if (!armed) return;                          // not over the geometry yet
  if (ev.buttons !== 0) return;               // don't slice while rotating
  const r = gd.getBoundingClientRect();
  frac = Math.min(1.0, Math.max(0.03, 1 - (ev.clientY - r.top) / r.height + CUT_LIFT));
  if (!pending) { pending = true; requestAnimationFrame(() => { applyCut(frac); pending = false; }); }
});
gd.addEventListener('mouseleave', () => { armed = false; applyCut(1.0); });
</script>
"""
    html = (html.replace("HEIGHT", str(height - 20))
                .replace("CAPSPEC", json.dumps(cap) if cap is not None else "null")
                .replace("SPEC", json.dumps(spec)))
    components.html(html, height=height)


# ---- Previews (second pass into the setup tabs) -----------------------------
with arr_preview:
    st.plotly_chart(render3d.array3d_figure(
        ring, shape_key, el_w_mm / 1000.0, el_h_mm / 1000.0,
        "Array and transducer elements (interactive 3D)"), width="stretch")
    ap = (f"{el_shape.lower()} {el_w_mm:.1f}x{el_h_mm:.1f} mm "
          f"(~{max(len(i) for i, _ in footprints)} grid pts)"
          if footprints is not None else "point")
    st.write(f"3D cylinder | {n_elem} elements ({ap}) | grid {n}x{n}x{n} "
             f"(h={h * 1e3:.2f} mm) | dt {dt * 1e9:.1f} ns | {len(src_list)} transmits used")
    if footprints is not None:
        pts = [len(i) for i, _ in footprints]
        st.caption(f"Element footprint sampled onto {min(pts)}-{max(pts)} grid points "
                   "per element; the aperture is simulated on transmit and receive "
                   "in every solver (acoustic and elastic).")

with exc_preview:
    import plotly.graph_objects as go

    def _line_fig(title, xtitle, ytitle=""):
        f = go.Figure()
        f.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle,
                        height=280, margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", y=-0.25))
        return f

    cw1, cw2 = st.columns(2)
    with cw1:
        n_show = len(exc_clk) + 4 * half_period
        t_clk = np.arange(n_show) / clk_hz * 1e6
        drive = np.zeros(n_show); drive[:len(exc_clk)] = exc_clk
        fig = _line_fig("FPGA drive (discrete levels at clock rate)", "us")
        fig.add_trace(go.Scatter(x=t_clk, y=drive, mode="lines",
                                 line=dict(shape="hv", width=1.5), name="drive"))
        fig.update_yaxes(range=[-1.3, 1.3])
        st.plotly_chart(fig, width="stretch")

        spec = np.abs(np.fft.rfft(wavelet_eff)); freqs = np.fft.rfftfreq(nt, dt) / 1e6
        fig = _line_fig("Output spectrum", "MHz", "normalised")
        fig.add_trace(go.Scatter(x=freqs, y=spec / (spec.max() + 1e-30),
                                 mode="lines", name="spectrum"))
        fig.update_xaxes(range=[0, f0_mhz * 4])
        st.plotly_chart(fig, width="stretch")
    with cw2:
        t_us = np.arange(nt) * dt * 1e6
        fig = _line_fig("Acoustic output", "us")
        fig.add_trace(go.Scatter(x=t_us, y=wavelet, mode="lines", name="acoustic (TX)",
                                 line=dict(width=1), opacity=0.55))
        fig.add_trace(go.Scatter(x=t_us, y=wavelet_eff, mode="lines",
                                 name="after RX filter", line=dict(width=1.8)))
        st.plotly_chart(fig, width="stretch")
    st.write(f"Chain: {sig_type.split(' (')[0]} at {clk_mhz} MHz clock "
             f"({len(exc_clk)} clocks) -> "
             + (f"TX low-pass {tx_cut_x:.2f}xf0 -> " if tx_filt_on else "no TX filter -> ")
             + f"transducer (f0 {f0_mhz} MHz, bw {frac_bw:.2f}) -> "
             + (f"RX band-pass {rx_lo_x:.2f}-{rx_hi_x:.1f}xf0." if rx_filt_on else "no RX filter."))
    st.caption("The drive is the exact digital pattern the tx_pulser RTL generates "
               "(verified bit-exact against tx_pulser.sv); the RX filter is applied "
               "to every received channel and to the wavelet used for reconstruction.")

with smp_preview:
    if poly:
        present = len(np.unique(labels[labels >= 0]))
        grain_colat_deg = np.degrees(np.arccos(np.clip(axes3[:, 2], -1.0, 1.0)))
        fig3d = render3d.polycrystal_figure(
            labels, grain_colat_deg, h,
            "3D Voronoi polycrystal, grains coloured by c-axis colatitude (interactive)",
            vmin=0.0, vmax=90.0, melt_mask=melt_mask)
        render3d.add_elements_plotly(fig3d, elem_abs)
        cap_vol = labels.copy()
        if melt_mask is not None:
            cap_vol[melt_mask] = -2                      # fluid pocket on the cut face
        cap_colors = {str(k): c for k, c in
                      render3d.grain_colors(grain_colat_deg, 0.0, 90.0).items()}
        cap_colors["-2"] = "rgb(48,96,192)"
        plotly_hover_slice(fig3d, cap=dict(vol=cap_vol.astype(int).tolist(),
                                           h_mm=h * 1e3, colors=cap_colors))
        st.caption("Move the cursor up/down over the chart to slice the polycrystal "
                   "horizontally (the cut face stays closed, like a solid section); "
                   "move the cursor away to restore the full geometry.")
        st.write(f"3D Voronoi polycrystal: {present} grains of single-crystal "
                 f"{mat_label}, each with its own 3D c-axis"
                 + (", with a fluid pocket" if melt_on else "") + ".")
        st.caption("Acquisition runs the FULL 3D anisotropic elastic solver: each "
                   "grain carries its complete rotated stiffness tensor "
                   "(all 21 components), propagating qP and both qS waves with "
                   "mode conversion. The apparent-qP map is the display / "
                   "reconstruction reference. Cyan dots mark the elements.")
    else:
        fig3d = render3d.plotly_figure(
            c_true, h, c_coup, c_spec, flaw_c if flaw_on else None,
            "Sample in the array (interactive 3D)")
        render3d.add_elements_plotly(fig3d, elem_abs)
        plotly_hover_slice(fig3d)
        st.caption("Cyan dots mark the transducer element positions around the sample. "
                   "Move the cursor up/down over the chart to slice the sample "
                   "horizontally; move it away to restore the full geometry.")


# ---- Pipeline tabs ----------------------------------------------------------
with tab_acq:
    if poly:
        st.caption("Polycrystal acquisition runs the full 3D anisotropic elastic "
                   "solver (21-component stiffness per grain) in Python; roughly "
                   "1-2 s per transmit at the default grid.")
    else:
        st.caption("3D acquisition and inversion are heavier; expect this to take a little while.")
    if st.button("Run acquisition", type="primary"):
        acq_bar = st.progress(0.0, text="Simulating full-matrix capture ...")

        def _acq_progress(done, total, elapsed):
            eta = elapsed / done * (total - done)
            acq_bar.progress(done / total,
                             text=f"Acquisition: transmit {done}/{total} "
                                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        data = run_acquisition(phantom.velocity_to_m(c_true), progress=_acq_progress)
        acq_bar.empty()
        data = apply_rx_filter(data, axis=1)            # RX front end, every channel
        elem_pos = ring.element_positions
        st.session_state.ds = Dataset(
            geometry=ArrayGeometry(elem_pos, ring.radius_m, f0, "cylinder"),
            data=data, sample_rate_hz=1.0 / dt, tx_wavelet=wavelet_rec,
            tx_centre_freq_hz=f0, nominal_speed_m_s=c_coup,
            ground_truth={"c": c_true, "h_m": h})
        st.session_state.src_list = src_list
        st.session_state.wavelet_eff = wavelet_rec
        # FPGA capture stage: always part of the pipeline, not an option.
        if hw_cosim.available():
            with st.spinner("FPGA capture stage: streaming through rx_capture.sv (Icarus Verilog) ..."):
                n_frames = int(min(3, data.shape[0]))
                cap_nt = int(min(data.shape[1], 384))
                sub = data[:n_frames, :cap_nt, :].transpose(0, 2, 1)   # (frames, ch, samples)
                scale = 30000.0 / (np.max(np.abs(sub)) + 1e-30)
                q = np.round(sub * scale).astype(np.int64)
                rx_sv = os.path.join(HERE, "..", "..", "hardware", "fpga", "rtl", "rx_capture.sv")
                try:
                    rtl = hw_cosim.run_rx_capture(q, rx_sv)
                    st.session_state.hw = (rtl, bool(np.array_equal(rtl, q)), n_frames, cap_nt)
                except Exception as e:
                    st.session_state.hw = None
                    st.error(f"FPGA capture stage failed: {e}")
        else:
            st.session_state.hw = None
            st.warning("Icarus Verilog (iverilog) not found: the FPGA capture stage "
                       "of the pipeline was skipped.")
        st.success(f"Acquired {data.shape[0]} transmits x {data.shape[1]} samples x {data.shape[2]} channels"
                   + (" | RX band-pass applied" if rx_filt_on else ""))

    if "ds" in st.session_state:
        ds = st.session_state.ds
        import plotly.graph_objects as go
        from scipy.signal import hilbert
        cda, cdb, cdc = st.columns([1, 1, 1])
        with cda:
            tx_sel = (st.slider("Transmit to display", 0, ds.n_tx - 1, 0)
                      if ds.n_tx > 1 else 0)
        with cdb:
            norm_per_ch = st.radio("Normalisation", ["Per channel", "Global"],
                                   horizontal=True,
                                   help="Per channel makes every element's arrival "
                                        "visible; Global shows true relative amplitude.")
        with cdc:
            dyn = st.slider("Dynamic range (dB)", 20, 60, 40, 5)
        d = ds.data[tx_sel].T                               # (channel, sample)
        env = np.abs(hilbert(d, axis=1))
        ref = (env.max(axis=1, keepdims=True) if norm_per_ch == "Per channel"
               else env.max())
        env_db = 20.0 * np.log10(env / (ref + 1e-30) + 1e-6)
        t_us = np.arange(ds.n_samples) / ds.sample_rate_hz * 1e6
        fig = go.Figure(go.Heatmap(
            z=env_db, x=t_us, y=np.arange(ds.n_rx),
            colorscale="Inferno", zmin=-dyn, zmax=0,
            colorbar=dict(title="dB"),
            hovertemplate="t %{x:.2f} us | rx %{y} | %{z:.1f} dB<extra></extra>"))
        fig.update_layout(
            title=f"Received frame (transmit {tx_sel}), envelope [dB]",
            xaxis_title="time (us)", yaxis_title="receive element",
            height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each row is one receiving element; bright ridges are wave "
                   "arrivals. The earliest, strongest rows are the elements next "
                   "to the transmitter; the arrival time grows with distance "
                   "around the array (the sloping moveout). Hover for exact values.")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            path = tf.name
        to_uarp_set(ds, path)
        with open(path, "rb") as f:
            st.download_button("Download UARP/UDSP file", f.read(), "acquisition_udsp.h5")

        st.divider()
        st.subheader("FPGA capture stage (rx_capture RTL)")
        if st.session_state.get("hw"):
            rtl, match, nf, cnt = st.session_state.hw
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("RTL matches acquisition", "bit-exact" if match else "MISMATCH")
                st.caption(f"{nf} transmits x {cnt} samples streamed through "
                           "rx_capture.sv in Icarus Verilog as part of the acquisition. "
                           "Bit-exact means every integer sample out of the RTL equals "
                           "the simulated acquisition after ADC quantisation.")
            with c2:
                t_us_hw = np.arange(cnt) / ds.sample_rate_hz * 1e6
                vmax_hw = float(np.abs(rtl[0]).max()) + 1e-30
                fig = go.Figure(go.Heatmap(
                    z=rtl[0], x=t_us_hw, y=np.arange(rtl.shape[1]),
                    colorscale="RdBu", zmin=-vmax_hw, zmax=vmax_hw,
                    colorbar=dict(title="ADC counts"),
                    hovertemplate="t %{x:.2f} us | ch %{y} | %{z} counts<extra></extra>"))
                fig.update_layout(
                    title="FPGA RTL-captured frame (transmit 0), quantised amplitude",
                    xaxis_title="time (us)", yaxis_title="capture channel",
                    height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("The FPGA capture stage runs automatically with every acquisition "
                    "(requires Icarus Verilog).")
    else:
        st.info("Run an acquisition to see the received data.")

with tab_img:
    if "ds" not in st.session_state:
        st.info("Run an acquisition first.")
    else:
        npix = st.slider("Image pixels per axis", 40, 160, 50, 10)
        with st.spinner("Total focusing method ..."):
            img, _ = imaging.tfm(st.session_state.ds, npix=npix, half_size=spec_r_m)
        show_fig(show_model(img, "TFM", cmap="inferno"))
        st.caption("TFM assumes a constant speed; it images reflectivity, not sound speed.")

with tab_fwi:
    if "ds" not in st.session_state:
        st.info("Run an acquisition first.")
    else:
        n_iter = st.slider("FWI iterations", 4, 16, 8 if poly else 6)
        _MISFITS = {
            "L2 (least squares)": "l2",
            "GCN (normalised cross-correlation)": "gcn",
            "Envelope (phase-insensitive, unmodelled-physics robust)": "envelope",
            "Envelope-GCN (scale- and phase-insensitive)": "egcn",
            "Traveltime (cross-correlation lags, kinematic)": "traveltime",
            "Graph-space OT (optimal transport, cycle-skip robust)": "gsot",
        }
        misfit_choice = st.selectbox(
            "Misfit functional", list(_MISFITS),
            index=1 if poly else 0,
            help="L2/GCN compare waveforms (C++ accelerated). Envelope and "
                 "Envelope-GCN compare Hilbert envelopes — robust when the "
                 "operator cannot model the waveform physics (anisotropy, "
                 "elasticity). Traveltime penalises per-trace arrival lags — "
                 "the purely kinematic observable that anisotropic wave speeds "
                 "perturb. The robust misfits run on the Python core (slower).")
        mtype = _MISFITS[misfit_choice]
        if mtype == "gcn":
            st.caption("GCN widens the convergence basin against cycle skipping; "
                       "amplitude-insensitive per trace (C++ accelerated).")
        elif mtype in ("envelope", "egcn", "traveltime", "gsot"):
            st.caption("Robust misfit: adjoint source verified by finite-difference "
                       "gradient check; runs on the Python core, so iterations are "
                       "slower than L2/GCN."
                       + (" GSOT additionally solves an optimal assignment per trace "
                          "(heaviest, most convex)." if mtype == "gsot" else ""))
        if poly:
            st.caption("The data are full 3D anisotropic elastic; acoustic FWI "
                       "reconstructs an APPARENT velocity field (compare against the "
                       "per-grain apparent-qP reference) — the model-mismatch question "
                       "this platform is built to study. GCN misfit and a smoothed "
                       "gradient are the defaults here: they suppress the source-imprint "
                       "artefacts the elastic/acoustic mismatch otherwise produces.")
        if st.button("Run FWI", type="primary"):
            ds = st.session_state.ds
            coords = np.mgrid[0:n, 0:n, 0:n].astype(float) * h
            cc = (n - 1) * h / 2
            r = np.sqrt((coords[1] - cc) ** 2 + (coords[2] - cc) ** 2)
            mask = (r <= spec_r_m * (0.85 if poly else 0.95)).astype(float)
            if poly:
                hi_v = max(float(c_true.max()), c_coup) * 1.06
                lo_v = min(float(c_true[c_true > 0].min()), c_coup) * 0.9
                m_bounds = (phantom.velocity_to_m(hi_v), phantom.velocity_to_m(lo_v))
            else:
                m_bounds = (phantom.velocity_to_m(c_spec + 700), phantom.velocity_to_m(min(flaw_c, c_spec) - 500))
            wav_rec = st.session_state.get("wavelet_eff", wavelet_rec)
            fwi_bar = st.progress(0.0, text=f"FWI: starting ({n_iter} iterations, {mtype.upper()}) ...")
            t0_fwi = time.time()

            def _fwi_progress(done, total):
                elapsed = time.time() - t0_fwi
                eta = elapsed / done * (total - done)
                fwi_bar.progress(done / total,
                                 text=f"FWI: iteration {done}/{total} "
                                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            use_cpp = BACKEND is not fwi and footprints is None
            backend = BACKEND if use_cpp else None
            m_rec, hist = fwi.invert(
                phantom.velocity_to_m(c_bg), ring, wav_rec, dt, h, nt, ds.data,
                src_list=st.session_state.get("src_list"), n_iter=n_iter, step_frac=0.03,
                update_mask=mask, m_bounds=m_bounds, backend=backend, misfit_type=mtype,
                footprints=footprints, progress=_fwi_progress,
                smooth_sigma=2.0 if poly else 1.0)
            fwi_bar.empty()
            st.session_state.rec = (phantom.m_to_velocity(m_rec), hist)
        if "rec" in st.session_state:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            c_rec, hist = st.session_state.rec
            mm = np.arange(n) * h * 1e3

            def slice_fig(vol, title):
                fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                                    subplot_titles=["xy (mid z)", "xz (mid y)", "yz (mid x)"])
                views = [(vol[n // 2], (0, 1)), (vol[:, n // 2, :], (0, 2)),
                         (vol[:, :, n // 2], (1, 2))]
                p = elem_abs * 1e3
                for col, (sl, cols) in enumerate(views, start=1):
                    fig.add_trace(go.Heatmap(
                        z=sl, x=mm, y=mm, coloraxis="coloraxis",
                        hovertemplate="%{x:.1f}, %{y:.1f} mm | %{z:.0f} m/s<extra></extra>"),
                        1, col)
                    fig.add_trace(go.Scatter(
                        x=p[:, cols[0]], y=p[:, cols[1]], mode="markers",
                        marker=dict(size=6, symbol="circle-open", color="#00e5ff",
                                    line=dict(width=1.2)),
                        showlegend=False, hoverinfo="skip"), 1, col)
                fig.update_layout(
                    title=title, height=330, margin=dict(l=10, r=10, t=60, b=10),
                    coloraxis=dict(colorscale="Viridis", cmin=vlo, cmax=vhi,
                                   colorbar=dict(title="m/s")))
                return fig

            def volume_fig(vol, title):
                zc, yc, xc = np.meshgrid(mm, mm, mm, indexing="ij")
                fig = go.Figure(go.Volume(
                    x=xc.ravel(), y=yc.ravel(), z=zc.ravel(), value=vol.ravel(),
                    isomin=vlo, isomax=vhi, opacity=0.08, surface_count=14,
                    colorscale="Viridis", colorbar=dict(title="m/s"),
                    caps=dict(x_show=False, y_show=False, z_show=False)))
                fig.add_trace(go.Scatter3d(
                    x=elem_abs[:, 0] * 1e3, y=elem_abs[:, 1] * 1e3, z=elem_abs[:, 2] * 1e3,
                    mode="markers", marker=dict(size=3, color="#00e5ff"),
                    showlegend=False))
                fig.update_layout(title=title, height=430,
                                  margin=dict(l=0, r=0, t=40, b=0),
                                  scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                                             zaxis_title="z (mm)"))
                return fig

            true_title = "Apparent qP speed (true)" if poly else "True model"
            st.plotly_chart(slice_fig(c_true, true_title), width="stretch")
            st.plotly_chart(slice_fig(c_rec, "FWI reconstruction"), width="stretch")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(volume_fig(c_true, f"{true_title} (3D volume)"),
                                width="stretch")
            with cc2:
                st.plotly_chart(volume_fig(c_rec, "FWI reconstruction (3D volume)"),
                                width="stretch")
            st.caption("3D volume views: colour and opacity follow sound speed, so the "
                       "couplant is transparent and the specimen structure is visible. "
                       "Rotate and zoom; hover the slices for exact speeds.")
            figm = go.Figure(go.Scatter(y=np.array(hist) / hist[0],
                                        mode="lines+markers", name="misfit"))
            figm.update_yaxes(type="log", title="relative misfit")
            figm.update_layout(title="Misfit convergence", xaxis_title="iteration",
                               height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figm, width="stretch")
            st.success(f"Misfit reduced to {hist[-1]/hist[0]*100:.1f}% of initial")
