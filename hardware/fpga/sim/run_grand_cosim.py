"""Grand end-to-end co-simulation: the whole OpenUSCT loop, RTL at both ends.

  1. tx_pulser RTL generates the transmit excitation.
  2. transducer model -> acoustic wavelet.
  3. OpenUSCT propagates it for every transmit -> received full-matrix data.
  4. acq_stream_top RTL sequences, captures, and AXI-Streams the frames out.
  5. the streamed frames are checked against the system, then written to the
     UARP/UDSP format and reloaded.
  6. FWI reconstructs the flaw from the RTL-captured data.

Run:  python run_grand_cosim.py   (needs iverilog + vvp on PATH)
Output: grand.png
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "simulation"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "software", "python"))

from ringfwi import fwi, phantom, transducer
from ringfwi.dataset import Dataset
from ringfwi.geometry import RingArray
from ringfwi.uarp_format import from_uarp_set, to_uarp_set
import uap  # C++ backend for FWI speed

# Match tb_grand.sv
N_ELEM = N_CH = 12
LEN = TOTAL = 280
CLK_HZ = 100e6
HALF, NHC, DEAD = 100, 6, 8          # 0.5 MHz excitation
PULSER = os.path.join(HERE, "..", "rtl", "tx_pulser.sv")
PULSER_TB = os.path.join(HERE, "..", "tb", "tb_tx_pulser.sv")
ACQ_RTL = [os.path.join(HERE, "..", "rtl", f) for f in
           ("rx_capture.sv", "tx_sequencer.sv", "axi_stream_out.sv", "acq_stream_top.sv")]
ACQ_TB = os.path.join(HERE, "..", "tb", "tb_grand.sv")


def run_pulser():
    subprocess.run(["iverilog", "-g2012", "-o", "pulser.out", PULSER, PULSER_TB], check=True)
    subprocess.run(["vvp", "pulser.out", f"+HALF={HALF}", f"+NHC={NHC}", f"+DEAD={DEAD}"], check=True)
    p, n = [], []
    with open("pulser_out.txt") as fp:
        for line in fp:
            if line.strip():
                a, b = line.split(); p.append(int(a)); n.append(int(b))
    return np.array(p) - np.array(n)


def main():
    os.chdir(HERE)

    # 1-2. FPGA pulser -> transducer -> acoustic wavelet.
    exc = run_pulser()
    fc = CLK_HZ / (2 * HALF)                       # 0.5 MHz
    dt = 6.0e-8
    fs = 1.0 / dt
    wavelet = transducer.acoustic_wavelet(exc, CLK_HZ, fs, fc, 0.6, n_out=LEN)
    print(f"[1] FPGA transmit wavelet at {fc/1e6:.2f} MHz")

    # 3. Physics: specimen with a flaw, full-matrix capture with that wavelet.
    h = 3.0e-4
    ring = RingArray(n_elements=N_ELEM, radius_m=0.009, domain_m=0.024, h=h)
    ng = ring.n
    c_bg = phantom.coupling_background((ng, ng), 3000.0, 1500.0, 0.007, h)
    c_true = phantom.add_inclusion(c_bg, (0.58, 0.50), 0.0025, 2600.0, h)
    dobs = uap.forward_fmc(phantom.velocity_to_m(c_true), ring, wavelet, dt, h, LEN)
    print(f"[3] simulated {N_ELEM}x{N_CH} FMC on a {ng}x{ng} grid")

    # 4. RTL acquisition: quantise, stream through the hardware, collect beats.
    wav = dobs.transpose(0, 2, 1)                   # (frame, channel, time)
    scale = 30000.0 / (np.max(np.abs(wav)) + 1e-30)
    qwav = np.round(wav * scale).astype(np.int64)
    with open("acq_samples.hex", "w") as fp:
        for f in range(N_ELEM):
            for ch in range(N_CH):
                for t in range(TOTAL):
                    fp.write(format(int(qwav[f, ch, t]) & 0xFFFF, "04x") + "\n")

    subprocess.run(["iverilog", "-g2012", "-o", "grand.out", *ACQ_RTL, ACQ_TB], check=True)
    subprocess.run(["vvp", "grand.out"], check=True)
    beats = []
    with open("stream_out.txt") as fp:
        for line in fp:
            if line.strip():
                beats.append(int(line.split()[0]))
    rtl = np.array(beats, dtype=np.int64).reshape(N_ELEM, N_CH, LEN)
    rtl_ok = np.array_equal(rtl, qwav)
    print(f"[4] RTL acquisition streamed {len(beats)} beats; matches system: {'YES' if rtl_ok else 'NO'}")

    # 5. Write the RTL-captured data to the UARP format and reload.
    rtl_fmc = (rtl.transpose(0, 2, 1) / scale)      # (n_tx, nt, n_rx) float
    ds = Dataset(geometry=ring_geometry(ring, h), data=rtl_fmc, sample_rate_hz=fs,
                 tx_wavelet=wavelet, tx_centre_freq_hz=fc, nominal_speed_m_s=1500.0)
    to_uarp_set(ds, "grand_udsp.h5")
    reloaded = from_uarp_set("grand_udsp.h5")
    uarp_ok = np.allclose(reloaded.data, rtl_fmc, rtol=1e-4, atol=1e-9)
    print(f"[5] written to UARP/UDSP and reloaded: {'YES' if uarp_ok else 'NO'}")

    # 6. FWI reconstruction, reading the data back from the UARP file.
    dobs_uarp = reloaded.data                       # loaded from grand_udsp.h5
    yy, xx = np.mgrid[0:ng, 0:ng].astype(float) * h
    cc = (ng - 1) * h / 2
    mask = (np.hypot(xx - cc, yy - cc) <= 0.007 * 0.95).astype(float)
    m_bounds = (phantom.velocity_to_m(3600.0), phantom.velocity_to_m(2400.0))
    m_rec, hist = fwi.invert(phantom.velocity_to_m(c_bg), ring, wavelet, dt, h, LEN, dobs_uarp,
                             n_iter=10, step_frac=0.04, update_mask=mask,
                             m_bounds=m_bounds, backend=uap)
    c_rec = phantom.m_to_velocity(m_rec)
    print(f"[6] FWI misfit {hist[0]:.3e} -> {hist[-1]:.3e} ({hist[-1]/hist[0]*100:.1f}%)")

    _figure(wavelet, dt, fc, rtl, c_true, c_rec, hist, h, ng)
    ok = rtl_ok and uarp_ok and hist[-1] < 0.5 * hist[0]
    print("\nPASS: whole platform loop verified, RTL at both ends" if ok else "\nFAIL")
    return 0 if ok else 1


def ring_geometry(ring, h):
    from ringfwi.dataset import ArrayGeometry
    return ArrayGeometry(element_pos=ring.element_positions, radius_m=ring.radius_m,
                         centre_freq_hz=0.0, array_type="ring")


def _figure(wavelet, dt, fc, rtl, c_true, c_rec, hist, h, ng):
    nt = len(wavelet)
    spec = np.abs(np.fft.rfft(wavelet)); freqs = np.fft.rfftfreq(nt, dt) / 1e6
    ext = [0, (ng - 1) * h * 1e3, 0, (ng - 1) * h * 1e3]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(np.arange(nt) * dt * 1e6, wavelet); ax[0, 0].set_title("FPGA transmit wavelet"); ax[0, 0].set_xlabel("us")
    ax[0, 1].plot(freqs, spec / spec.max()); ax[0, 1].axvline(fc / 1e6, color="r", ls="--")
    ax[0, 1].set_xlim(0, 3); ax[0, 1].set_title("Wavelet spectrum"); ax[0, 1].set_xlabel("MHz")
    ax[0, 2].imshow(rtl[0].astype(float), aspect="auto", cmap="gray"); ax[0, 2].set_title("RTL-captured frame (tx 0)")
    ax[0, 2].set_xlabel("sample"); ax[0, 2].set_ylabel("channel")
    for a, field, ttl in ((ax[1, 0], c_true, "True model"), (ax[1, 1], c_rec, "FWI (from RTL data)")):
        im = a.imshow(field, origin="lower", extent=ext, cmap="viridis", vmin=2500, vmax=3100)
        a.set_title(ttl); a.set_xlabel("mm"); fig.colorbar(im, ax=a, fraction=0.046)
    ax[1, 2].semilogy(np.array(hist) / hist[0], "o-"); ax[1, 2].set_title("FWI misfit"); ax[1, 2].grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "grand.png"), dpi=120)
    print(f"saved {os.path.join(HERE, 'grand.png')}")


if __name__ == "__main__":
    sys.exit(main())
