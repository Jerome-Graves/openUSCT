"""Run the FPGA capture RTL on acquisition data, from the GUI.

Generates a testbench sized to the current acquisition, feeds the (quantised)
received channel data through the ``rx_capture`` SystemVerilog module in Icarus
Verilog, and returns the RTL-captured frames. This lets the GUI show the real
FPGA capture datapath processing the simulated acquisition.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

_TB = """`timescale 1ns/1ps
module tb_gui;
  localparam int N_CH={n_ch};
  localparam int DEPTH={depth};
  localparam int SAMPLE_W=16;
  localparam int ADDR_W=$clog2(DEPTH);
  localparam int LEN={nt};
  localparam int TOTAL={nt};
  localparam int NFRAMES={n_frames};
  logic clk=0,rst_n=0; always #5 clk=~clk;
  logic trigger; logic [ADDR_W-1:0] acq_delay, acq_len;
  logic adc_valid; logic signed [SAMPLE_W-1:0] adc_data [0:N_CH-1];
  logic done; logic [ADDR_W-1:0] rd_addr; logic [$clog2(N_CH)-1:0] rd_ch;
  logic signed [SAMPLE_W-1:0] rd_data;
  rx_capture #(.N_CH(N_CH),.DEPTH(DEPTH),.SAMPLE_W(SAMPLE_W)) dut
    (.clk,.rst_n,.trigger,.acq_delay,.acq_len,.adc_valid,.adc_data,.done,.rd_addr,.rd_ch,.rd_data);
  logic signed [SAMPLE_W-1:0] smem [0:NFRAMES*N_CH*TOTAL-1];
  integer fout,f,t,ch;
  initial begin
    $readmemh("gui_samples.hex", smem);
    fout=$fopen("gui_out.txt","w");
    trigger=0; adc_valid=0; acq_len=LEN[ADDR_W-1:0]; acq_delay='0;
    for(ch=0;ch<N_CH;ch++) adc_data[ch]='0;
    repeat(3)@(posedge clk); rst_n=1; @(posedge clk);
    for(f=0;f<NFRAMES;f++) begin
      trigger=1; @(posedge clk); trigger=0; @(posedge clk);
      for(t=0;t<TOTAL;t++) begin
        adc_valid=1;
        for(ch=0;ch<N_CH;ch++) adc_data[ch]=smem[f*N_CH*TOTAL+ch*TOTAL+t];
        @(posedge clk);
      end
      adc_valid=0;
      while(!done) @(posedge clk);
      for(ch=0;ch<N_CH;ch++) for(t=0;t<LEN;t++) begin
        rd_ch=ch[$clog2(N_CH)-1:0]; rd_addr=t[ADDR_W-1:0];
        @(posedge clk); @(posedge clk); $fdisplay(fout,"%0d",rd_data);
      end
    end
    $fclose(fout); $finish;
  end
endmodule
"""


def _tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(r"C:\iverilog\bin", name + ".exe")
    return cand if os.path.exists(cand) else None


def available():
    return _tool("iverilog") is not None and _tool("vvp") is not None


def run_rx_capture(frames_ch_t, rx_capture_sv):
    """frames_ch_t: (n_frames, n_ch, nt) int array -> RTL-captured (same shape)."""
    iverilog, vvp = _tool("iverilog"), _tool("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog (iverilog/vvp) not found")

    rx_capture_sv = os.path.abspath(rx_capture_sv)
    n_frames, n_ch, nt = frames_ch_t.shape
    work = tempfile.mkdtemp(prefix="uap_hwcosim_")
    with open(os.path.join(work, "gui_samples.hex"), "w") as f:
        for fr in range(n_frames):
            for ch in range(n_ch):
                for t in range(nt):
                    f.write(format(int(frames_ch_t[fr, ch, t]) & 0xFFFF, "04x") + "\n")

    with open(os.path.join(work, "tb_gui.sv"), "w") as f:
        f.write(_TB.format(n_ch=n_ch, depth=nt, nt=nt, n_frames=n_frames))

    out = os.path.join(work, "gui.out")
    subprocess.run([iverilog, "-g2012", "-o", out, rx_capture_sv, "tb_gui.sv"],
                   cwd=work, check=True, capture_output=True, text=True)
    subprocess.run([vvp, out], cwd=work, check=True, capture_output=True, text=True)

    vals = []
    with open(os.path.join(work, "gui_out.txt")) as f:
        for line in f:
            if line.strip():
                vals.append(int(line))
    return np.array(vals, dtype=np.int64).reshape(n_frames, n_ch, nt)
