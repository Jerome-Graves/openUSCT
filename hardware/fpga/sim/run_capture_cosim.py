"""System co-simulation: rx_capture RTL vs the OpenUSCT acquisition.

Simulates transmit events with the OpenUSCT forward solver, quantises the
received channel waveforms to ADC samples, streams them through the SystemVerilog
capture module (with a per-frame acquisition delay), and checks the captured
frames bit-for-bit against the windowed system data. The frame with zero delay
must reproduce the OpenUSCT acquisition exactly: the FPGA capture path works with
the system.

Run:  python run_capture_cosim.py   (needs iverilog + vvp on PATH)
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "simulation"))

from ringfwi import phantom
from ringfwi.acquire import simulate_dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker

# Must match tb_rx_capture.sv
N_CH, TOTAL, LEN, NFRAMES = 8, 64, 48, 4
DELAYS = [0, 5, 10, 3]

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "..", "rtl", "rx_capture.sv")
TB = os.path.join(HERE, "..", "tb", "tb_rx_capture.sv")


def main():
    # Simulate an acquisition: 8-element ring, NFRAMES transmits, TOTAL samples.
    h = 1.0e-3
    ring = RingArray(n_elements=N_CH, radius_m=0.016, domain_m=0.040, h=h)
    dt = 8.0e-8
    wavelet = ricker(TOTAL, dt, 0.4e6)
    n = ring.n
    c = np.full((n, n), 1500.0)
    c = phantom.add_inclusion(c, (0.6, 0.5), 0.003, 2200.0, h)
    ds = simulate_dataset(c, ring, wavelet, dt, nominal_speed_m_s=1500.0,
                          src_list=list(range(NFRAMES)))

    # ds.data is (NFRAMES, TOTAL, N_CH); the capture wants (frame, channel, time).
    wav = ds.data.transpose(0, 2, 1)  # (NFRAMES, N_CH, TOTAL)
    scale = 30000.0 / (np.max(np.abs(wav)) + 1e-30)
    qwav = np.round(wav * scale).astype(np.int64)

    os.chdir(HERE)
    with open("cap_samples.hex", "w") as fp:
        for f in range(NFRAMES):
            for ch in range(N_CH):
                for t in range(TOTAL):
                    fp.write(format(int(qwav[f, ch, t]) & 0xFFFF, "04x") + "\n")
    with open("cap_delays.hex", "w") as fp:
        for f in range(NFRAMES):
            fp.write(format(DELAYS[f], "02x") + "\n")

    # Golden: captured[f,ch,t] = qwav[f, ch, delay[f] + t].
    gold = np.zeros((NFRAMES, N_CH, LEN), dtype=np.int64)
    for f in range(NFRAMES):
        gold[f] = qwav[f, :, DELAYS[f]:DELAYS[f] + LEN]

    subprocess.run(["iverilog", "-g2012", "-o", "cap.out", RTL, TB], check=True)
    subprocess.run(["vvp", "cap.out"], check=True)

    with open("cap_out.txt") as fp:
        vals = [int(x) for x in fp.read().split()]
    rtl = np.array(vals, dtype=np.int64).reshape(NFRAMES, N_CH, LEN)

    ok = True
    for f in range(NFRAMES):
        match = np.array_equal(rtl[f], gold[f])
        ok &= match
        print(f"frame {f}: delay={DELAYS[f]:2d}  captured {rtl[f].shape}  "
              f"{'OK (bit-exact)' if match else 'FAIL'}")

    # Frame 0 has zero delay: the capture must equal the system acquisition.
    sys_match = np.array_equal(rtl[0].T, qwav[0, :, :LEN].T)
    print(f"\nframe 0 reproduces the OpenUSCT acquisition exactly: "
          f"{'YES' if sys_match else 'NO'}")

    if ok and sys_match:
        print("\nPASS: FPGA capture datapath verified against the system")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
