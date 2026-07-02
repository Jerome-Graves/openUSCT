"""Full FMC acquisition co-simulation: acq_top RTL vs the OpenUSCT acquisition.

The transmit sequencer fires every element in turn; for each, the received
channel waveforms (from OpenUSCT) are streamed through the capture datapath. The
assembled N-by-N frames are checked bit-for-bit against the system acquisition,
proving the full trigger -> capture -> frame cycle works end to end.

Run:  python run_acq_cosim.py   (needs iverilog + vvp on PATH)
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

# Must match tb_acq_top.sv
N_ELEM, N_CH, TOTAL, LEN = 8, 8, 48, 48

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "..", "rtl", f) for f in
       ("rx_capture.sv", "tx_sequencer.sv", "acq_top.sv")]
TB = os.path.join(HERE, "..", "tb", "tb_acq_top.sv")


def main():
    h = 1.0e-3
    ring = RingArray(n_elements=N_ELEM, radius_m=0.016, domain_m=0.040, h=h)
    dt = 8.0e-8
    wavelet = ricker(TOTAL, dt, 0.4e6)
    n = ring.n
    c = np.full((n, n), 1500.0)
    c = phantom.add_inclusion(c, (0.6, 0.5), 0.003, 2200.0, h)
    ds = simulate_dataset(c, ring, wavelet, dt, nominal_speed_m_s=1500.0)  # all elements transmit

    # (n_tx, TOTAL, n_rx) -> (frame, channel, time)
    wav = ds.data.transpose(0, 2, 1)
    scale = 30000.0 / (np.max(np.abs(wav)) + 1e-30)
    qwav = np.round(wav * scale).astype(np.int64)

    os.chdir(HERE)
    with open("acq_samples.hex", "w") as fp:
        for f in range(N_ELEM):
            for ch in range(N_CH):
                for t in range(TOTAL):
                    fp.write(format(int(qwav[f, ch, t]) & 0xFFFF, "04x") + "\n")

    subprocess.run(["iverilog", "-g2012", "-o", "acq.out", *RTL, TB], check=True)
    subprocess.run(["vvp", "acq.out"], check=True)

    with open("acq_out.txt") as fp:
        vals = [int(x) for x in fp.read().split()]
    rtl = np.array(vals, dtype=np.int64).reshape(N_ELEM, N_CH, LEN)

    match = np.array_equal(rtl, qwav[:, :, :LEN])
    for f in range(N_ELEM):
        fm = np.array_equal(rtl[f], qwav[f, :, :LEN])
        print(f"transmit {f}: frame {rtl[f].shape}  {'OK' if fm else 'FAIL'}")

    print(f"\nfull N-by-N FMC acquisition matches the system: {'YES' if match else 'NO'}")
    if match:
        print("\nPASS: full transmit -> capture -> frame acquisition verified against the system")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
