"""Level 1 co-simulation: das_beamformer RTL vs an integer golden model.

Generates random per-channel samples, delays, and apodisation weights, computes
the exact integer delay-and-sum in Python, runs the SystemVerilog module in
Icarus Verilog on the same stimulus, and checks the RTL output bit for bit.

Run:  python run_das_cosim.py   (needs iverilog + vvp on PATH)
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

# Must match tb_das_beamformer.sv
N_CH, DEPTH, NTRIALS, SHIFT = 8, 64, 8, 15

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "..", "rtl", "das_beamformer.sv")
TB = os.path.join(HERE, "..", "tb", "tb_das_beamformer.sv")


def h16(v):
    return format(int(v) & 0xFFFF, "04x")


def main():
    rng = np.random.default_rng(1234)
    samples = rng.integers(-30000, 30000, size=(N_CH, DEPTH), dtype=np.int64)
    delays = rng.integers(0, DEPTH, size=(NTRIALS, N_CH), dtype=np.int64)
    weights = rng.integers(-32768, 32767, size=(NTRIALS, N_CH), dtype=np.int64)

    # Stimulus files (in this working directory, where vvp runs).
    os.chdir(HERE)
    with open("samples.hex", "w") as f:
        for ch in range(N_CH):
            for a in range(DEPTH):
                f.write(h16(samples[ch, a]) + "\n")
    with open("delays.hex", "w") as f:
        for t in range(NTRIALS):
            for ch in range(N_CH):
                f.write(format(int(delays[t, ch]) & 0x3FF, "03x") + "\n")
    with open("weights.hex", "w") as f:
        for t in range(NTRIALS):
            for ch in range(N_CH):
                f.write(h16(weights[t, ch]) + "\n")

    # Integer golden model: acc = sum_ch sample[ch][delay[ch]] * weight[ch].
    gold_acc, gold_res = [], []
    for t in range(NTRIALS):
        acc = 0
        for ch in range(N_CH):
            acc += int(samples[ch, delays[t, ch]]) * int(weights[t, ch])
        res = (acc + (1 << (SHIFT - 1))) >> SHIFT  # arithmetic (floor) shift
        gold_acc.append(acc)
        gold_res.append(res)

    # Compile and run the RTL.
    subprocess.run(["iverilog", "-g2012", "-o", "das.out", RTL, TB], check=True)
    subprocess.run(["vvp", "das.out"], check=True)

    with open("out.txt") as f:
        lines = [ln.split() for ln in f if ln.strip()]
    rtl_acc = [int(a) for a, _ in lines]
    rtl_res = [int(r) for _, r in lines]

    ok = True
    for t in range(NTRIALS):
        a_ok = rtl_acc[t] == gold_acc[t]
        r_ok = rtl_res[t] == gold_res[t]
        ok &= a_ok and r_ok
        print(f"trial {t}: acc rtl={rtl_acc[t]:>14d} gold={gold_acc[t]:>14d} {'OK' if a_ok else 'FAIL'} | "
              f"result rtl={rtl_res[t]:>7d} gold={gold_res[t]:>7d} {'OK' if r_ok else 'FAIL'}")

    if ok:
        print(f"\nPASS: RTL beamformer matches the golden model bit-for-bit ({NTRIALS} focal points)")
        return 0
    print("\nFAIL: mismatch")
    return 1


if __name__ == "__main__":
    sys.exit(main())
