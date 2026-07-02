"""Closed transmit loop: FPGA pulser (RTL) -> transducer -> acoustic wave.

Runs the tx_pulser RTL to generate the bipolar gate excitation, passes it
through the transducer model to obtain the acoustic wavelet, and propagates that
wavelet in the OpenUSCT forward model to produce received channel data. The pulse
the digital logic emits is the pulse the physics launches.

Run:  python run_tx_loop_cosim.py   (needs iverilog + vvp on PATH)
Output: ../figures is not used; writes tx_loop.png in this directory.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "simulation"))

from ringfwi import fwi, phantom, transducer
from ringfwi.geometry import RingArray

HERE = os.path.dirname(os.path.abspath(__file__))
PULSER = os.path.join(HERE, "..", "rtl", "tx_pulser.sv")
PULSER_TB = os.path.join(HERE, "..", "tb", "tb_tx_pulser.sv")

CLK_HZ = 100e6
HALF, NHC, DEAD = 25, 6, 4          # 100 MHz / (2*25) = 2 MHz excitation


def run_pulser():
    os.chdir(HERE)
    subprocess.run(["iverilog", "-g2012", "-o", "pulser.out", PULSER, PULSER_TB], check=True)
    subprocess.run(["vvp", "pulser.out", f"+HALF={HALF}", f"+NHC={NHC}", f"+DEAD={DEAD}"], check=True)
    p, n = [], []
    with open("pulser_out.txt") as fp:
        for line in fp:
            if line.strip():
                a, b = line.split()
                p.append(int(a)); n.append(int(b))
    return np.array(p) - np.array(n)      # bipolar excitation


def main():
    exc = run_pulser()

    fc = CLK_HZ / (2 * HALF)              # 2 MHz
    frac_bw = 0.6
    dt = 4.0e-8
    fs = 1.0 / dt
    nt = 400
    wavelet = transducer.acoustic_wavelet(exc, CLK_HZ, fs, fc, frac_bw, n_out=nt)

    # Propagate the FPGA-derived wavelet in OpenUSCT.
    h = 1.2e-4
    ring = RingArray(n_elements=8, radius_m=0.006, domain_m=0.016, h=h)
    ng = ring.n
    c = np.full((ng, ng), 1500.0)
    m = phantom.velocity_to_m(c)
    data = fwi.forward_fmc(m, ring, wavelet, dt, h, nt, src_list=[0])
    rx = data[0][:, 4]                    # a receive channel opposite the source

    # Spectrum of the acoustic wavelet.
    spec = np.abs(np.fft.rfft(wavelet))
    freqs = np.fft.rfftfreq(nt, dt) / 1e6
    f_peak = freqs[int(np.argmax(spec))]

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].step(np.arange(len(exc)), exc, where="post")
    ax[0, 0].set_title("FPGA pulser excitation (RTL)")
    ax[0, 0].set_xlabel("clock"); ax[0, 0].set_ylabel("gate (p - n)")

    hir = transducer.transducer_impulse_response(fc, frac_bw, fs)
    ax[0, 1].plot(np.arange(len(hir)) / fs * 1e6, hir)
    ax[0, 1].set_title(f"Transducer response ({fc/1e6:.1f} MHz, {frac_bw*100:.0f}% BW)")
    ax[0, 1].set_xlabel("us")

    ax[0, 2].plot(np.arange(nt) * dt * 1e6, wavelet)
    ax[0, 2].set_title("Acoustic transmit wavelet")
    ax[0, 2].set_xlabel("us")

    ax[1, 0].plot(freqs, spec / spec.max())
    ax[1, 0].axvline(fc / 1e6, color="r", ls="--", label=f"{fc/1e6:.1f} MHz")
    ax[1, 0].set_xlim(0, 8); ax[1, 0].set_title("Wavelet spectrum")
    ax[1, 0].set_xlabel("MHz"); ax[1, 0].legend()

    ax[1, 1].plot(np.arange(nt) * dt * 1e6, rx)
    ax[1, 1].set_title("Received channel (opposite element)")
    ax[1, 1].set_xlabel("us")
    ax[1, 2].axis("off")

    fig.tight_layout()
    out = os.path.join(HERE, "tx_loop.png")
    fig.savefig(out, dpi=120)
    print(f"excitation {len(exc)} clocks at {CLK_HZ/1e6:.0f} MHz -> {fc/1e6:.2f} MHz")
    print(f"acoustic wavelet peak frequency: {f_peak:.2f} MHz (target {fc/1e6:.2f})")
    print(f"received signal peak amplitude: {np.max(np.abs(rx)):.3e}")
    print(f"saved {os.path.normpath(out)}")
    print("\nPASS: FPGA transmit pulse propagated through the physics")


if __name__ == "__main__":
    main()
