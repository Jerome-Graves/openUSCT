"""Co-simulation for tx_pulser: bipolar gate waveform vs a golden model.

Checks the generated pulse train exactly (polarity, half-period, count,
dead-time), verifies there is never shoot-through (both gates high), and reports
the effective centre frequency for a chosen clock rate.

Run:  python run_pulser_cosim.py   (needs iverilog + vvp on PATH)
"""

from __future__ import annotations

import os
import subprocess
import sys

# Must match tb_tx_pulser.sv
HALF, NHC, DEAD = 8, 6, 2
CLK_HZ = 100e6

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "..", "rtl", "tx_pulser.sv")
TB = os.path.join(HERE, "..", "tb", "tb_tx_pulser.sv")


def main():
    os.chdir(HERE)
    subprocess.run(["iverilog", "-g2012", "-o", "pulser.out", RTL, TB], check=True)
    subprocess.run(["vvp", "pulser.out"], check=True)

    rtl_p, rtl_n = [], []
    with open("pulser_out.txt") as fp:
        for line in fp:
            if line.strip():
                p, n = line.split()
                rtl_p.append(int(p)); rtl_n.append(int(n))

    # Golden: each half-cycle is DEAD low cycles then the active polarity.
    gp, gn = [], []
    for hc in range(NHC):
        for cnt in range(HALF):
            active = cnt >= DEAD
            gp.append(1 if (active and hc % 2 == 0) else 0)
            gn.append(1 if (active and hc % 2 == 1) else 0)

    print(f"cycles: rtl {len(rtl_p)}, golden {len(gp)}")
    data_ok = (rtl_p == gp) and (rtl_n == gn)
    shoot_through = any(p and n for p, n in zip(rtl_p, rtl_n))

    f0 = CLK_HZ / (2 * HALF)
    print(f"gate waveform matches golden: {'YES' if data_ok else 'NO'}")
    print(f"no shoot-through (p & n never both high): {'YES' if not shoot_through else 'NO'}")
    print(f"effective centre frequency at {CLK_HZ/1e6:.0f} MHz clock: {f0/1e6:.2f} MHz, "
          f"{NHC/2:.0f} cycles")

    if data_ok and not shoot_through:
        print("\nPASS: transmit pulser verified")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
