"""AXI-Stream acquisition co-simulation, and the bridge to the UARP file.

Runs the full acquisition subsystem (sequencer + capture + AXI-Stream output),
collects the streamed frame beats, checks them bit-for-bit against the system,
verifies tlast marks frame boundaries, and then writes the streamed data as a
UARP/UDSP v4.0 file, the path the FPGA output takes to the acquisition format.

Run:  python run_stream_cosim.py   (needs iverilog + vvp on PATH)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "simulation"))

from ringfwi import phantom
from ringfwi.acquire import simulate_dataset
from ringfwi.dataset import Dataset
from ringfwi.geometry import RingArray
from ringfwi.sources import ricker
from ringfwi.uarp_format import from_uarp_set, to_uarp_set

N_ELEM, N_CH, TOTAL, LEN = 8, 8, 48, 48
HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "..", "rtl", f) for f in
       ("rx_capture.sv", "tx_sequencer.sv", "axi_stream_out.sv", "acq_stream_top.sv")]
TB = os.path.join(HERE, "..", "tb", "tb_acq_stream.sv")


def main():
    h = 1.0e-3
    ring = RingArray(n_elements=N_ELEM, radius_m=0.016, domain_m=0.040, h=h)
    dt = 8.0e-8
    wavelet = ricker(TOTAL, dt, 0.4e6)
    n = ring.n
    c = np.full((n, n), 1500.0)
    c = phantom.add_inclusion(c, (0.6, 0.5), 0.003, 2200.0, h)
    ds = simulate_dataset(c, ring, wavelet, dt, nominal_speed_m_s=1500.0)

    wav = ds.data.transpose(0, 2, 1)                      # (frame, channel, time)
    scale = 30000.0 / (np.max(np.abs(wav)) + 1e-30)
    qwav = np.round(wav * scale).astype(np.int64)

    os.chdir(HERE)
    with open("acq_samples.hex", "w") as fp:
        for f in range(N_ELEM):
            for ch in range(N_CH):
                for t in range(TOTAL):
                    fp.write(format(int(qwav[f, ch, t]) & 0xFFFF, "04x") + "\n")

    subprocess.run(["iverilog", "-g2012", "-o", "stream.out", *RTL, TB], check=True)
    subprocess.run(["vvp", "stream.out"], check=True)

    beats, lasts = [], []
    with open("stream_out.txt") as fp:
        for line in fp:
            if line.strip():
                d, l = line.split()
                beats.append(int(d)); lasts.append(int(l))
    beats = np.array(beats, dtype=np.int64)

    expected = N_ELEM * N_CH * LEN
    print(f"collected {len(beats)} beats (expected {expected})")
    assert len(beats) == expected, "wrong beat count"

    rtl = beats.reshape(N_ELEM, N_CH, LEN)
    data_ok = np.array_equal(rtl, qwav[:, :, :LEN])

    # tlast must be high exactly on the last beat of each frame.
    lasts = np.array(lasts).reshape(N_ELEM, N_CH * LEN)
    tlast_ok = np.all(lasts[:, -1] == 1) and np.all(lasts[:, :-1] == 0)

    print(f"AXI-Stream frames match the system: {'YES' if data_ok else 'NO'}")
    print(f"tlast marks every frame boundary: {'YES' if tlast_ok else 'NO'}")

    # Bridge: write the streamed acquisition as a UARP/UDSP v4.0 file.
    stream_fmc = rtl.transpose(0, 2, 1).astype(float)     # (n_tx, nt, n_rx)
    streamed = Dataset(geometry=ds.geometry, data=stream_fmc, sample_rate_hz=1.0 / dt,
                       tx_wavelet=wavelet, tx_centre_freq_hz=ds.tx_centre_freq_hz,
                       nominal_speed_m_s=1500.0)
    with tempfile.TemporaryDirectory() as d:
        upath = os.path.join(d, "fpga_acq_udsp.h5")
        to_uarp_set(streamed, upath)
        reloaded = from_uarp_set(upath)
        bridge_ok = reloaded.data.shape == stream_fmc.shape and reloaded.n_tx == N_ELEM
    print(f"streamed frames written to UARP/UDSP format and reloaded: {'YES' if bridge_ok else 'NO'}")

    if data_ok and tlast_ok and bridge_ok:
        print("\nPASS: AXI-Stream acquisition verified and bridged to the UARP format")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
