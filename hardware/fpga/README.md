# FPGA

The digital logic for the acquisition board's SoC/FPGA (a Xilinx Zynq in the
[electronics](../electronics/) spec), developed and verified in simulation
against the OpenUSCT software before any board exists.

The target system is a **matrix of transducers in a ring doing full-matrix
capture, with no beamforming** (reconstruction is full waveform inversion on the
host). So the core FPGA job is **acquisition**: trigger a transmit, then digitise
and buffer the received channel data into frames. A receive beamformer is
included as an optional auxiliary block, but it is not on the FMC/FWI path.

## Co-simulation

The way to develop the FPGA safely without hardware is to drive the RTL with the
same data the system produces and check its output against a trusted reference:

```
OpenUSCT forward solver  ->  received channel waveforms (what the ADCs see)
                            -> quantise to samples -> stream into the RTL
                            -> RTL emits captured frames
                            -> check bit-for-bit against the system acquisition
```

## What is here

- **`rtl/tx_sequencer.sv`** — the FMC transmit sequencing FSM: fires each element
  in turn, triggers the capture, hands the frame off downstream, and advances.
- **`rtl/tx_pulser.sv`** — the transmit pulse-timing generator: on fire it drives
  the pulser gates with a bipolar square-wave burst (programmable frequency and
  cycle count) with dead-time to prevent shoot-through.
- **`rtl/rx_capture.sv`** — the receive capture datapath: on a transmit trigger
  it waits a programmable acquisition delay, then captures a fixed-length window
  of samples from every channel in parallel into per-channel frame buffers. Each
  captured frame plus its delay is exactly one UARP/UDSP `Frame`.
- **`rtl/acq_top.sv`** — the acquisition subsystem: sequencer + capture wired
  together, running a complete N-by-N full-matrix acquisition.
- **`rtl/axi_stream_out.sv`** — streams each captured frame out as an AXI-Stream
  (the interface a Zynq DMA reads), honouring `tready` backpressure with `tlast`
  at each frame boundary.
- **`rtl/acq_stream_top.sv`** — the acquisition subsystem with AXI-Stream frame
  output: sequence -> capture -> stream to host, the bridge to the UARP file.
- **`sim/run_stream_cosim.py`** — co-simulates the streaming acquisition, checks
  the beats against the system, and writes the streamed frames to the UARP/UDSP
  v4.0 format.
- **`sim/run_acq_cosim.py`** — the full-acquisition co-simulation: OpenUSCT
  simulates the transmits, the sequencer drives the trigger->capture cycle, and
  the assembled N-by-N frames are checked bit-for-bit against the system.
- **`sim/run_capture_cosim.py`** — a focused co-simulation of the capture
  datapath alone, across several acquisition delays.
- **`rtl/das_beamformer.sv`** (auxiliary) — a fixed-point delay-and-sum receive
  beamformer, verified bit-exact against an integer golden model via
  `sim/run_das_cosim.py`. Useful for a B-mode preview mode; not used by the
  FMC/FWI path.

## Verification (done)

```bash
cd sim
python run_grand_cosim.py       # WHOLE loop, RTL at both ends (see below)
python run_stream_cosim.py      # acquisition + AXI-Stream + UARP bridge vs system
python run_acq_cosim.py         # full N-by-N acquisition vs the system, bit-exact
python run_capture_cosim.py     # capture datapath vs the system, bit-exact
python run_pulser_cosim.py      # transmit pulser gate waveform vs golden model
python run_tx_loop_cosim.py     # FPGA pulser -> transducer -> propagated wave
python run_das_cosim.py         # auxiliary beamformer vs golden model, bit-exact
```

`run_tx_loop_cosim.py` closes the transmit path: it runs the `tx_pulser` RTL to
generate the bipolar excitation, passes it through a transducer model
(`ringfwi.transducer`) to obtain the acoustic wavelet, and propagates that
wavelet in the OpenUSCT forward model. The pulse the digital logic emits is the
pulse the physics launches, verified to peak at the intended centre frequency.

`run_grand_cosim.py` runs the **entire platform loop in one go, with RTL at both
ends**: the `tx_pulser` RTL generates the transmit pulse, the transducer model
and forward solver propagate it for every element, the `acq_stream_top` RTL
sequences / captures / AXI-Streams the received frames, those frames are checked
against the system and written to the UARP/UDSP file, and FWI reconstructs the
flaw from the RTL-captured data. Output: `grand.png`.

Results: the full sequencer + capture + AXI-Stream acquisition reproduces the
OpenUSCT N-by-N dataset exactly (with tready backpressure) and its frames write
straight into the UARP/UDSP format; the capture path reproduces it across
acquisition delays; the beamformer matches its golden model exactly.

## Toolchain

Icarus Verilog 12 (`iverilog -g2012`, `vvp`). No vendor tools needed for
simulation. The same RTL is intended to synthesise for the Zynq; timing closure
and resource use are a Vivado (synthesis-stage) concern, separate from this
functional verification.

## Roadmap

- Integrate the pulse timing into `acq_stream_top` (fire `tx_pulser` per element)
  and model the transducer response so the emitted pulse closes back into the
  OpenUSCT forward model.
- An ADC-interface wrapper (deserialise real ADC lanes into the sample stream).
- A Zynq block design (PS/PL split, DMA from the AXI-Stream to the ARM cores,
  Ethernet streaming), with the host assembling the UARP file.
- Synthesis and timing closure in Vivado for the target part.
