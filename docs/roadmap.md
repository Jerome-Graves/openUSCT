# Roadmap

Phased so that each stage produces something usable on its own, and so the
software and simulation pillars (pure design and code) run ahead of the hardware
pillar (which needs fabrication).

## Phase 0 — Simulation core (done)

- Dimension-general acoustic finite-difference forward solver in the
  squared-slowness form: **2D and 3D** from one code path.
- Full-matrix-capture acquisition on a planar ring (2D) and a cylindrical array
  of stacked rings (3D).
- Exact discrete adjoint-state FWI gradient, verified against finite differences
  to machine precision in **both 2D and 3D**.
- End-to-end reconstruction demonstrations (2D flaw, 3D spherical flaw) and
  wavefield visualisation.
- Portable HDF5 `Dataset` format (2D and 3D) with round-trip tests.

Location: `simulation/`.

## Phase 1 — Software backbone (buildable now)

- Define the HDF5 `Dataset` / `ArrayGeometry` / `Image` schema (`software/schema/`).
- Python reference library reading and writing that schema.
- Delay-and-sum and total-focusing-method imaging on FMC data.
- The `Algorithm` plugin interface and one example plugin.
- The HAL abstraction with a simulation backend.

Deliverable: a user can simulate a dataset, save it, and image it with either a
built-in or a custom algorithm, entirely in Python.

## Phase 2 — C++ core and bindings (done)

- `libuap` header-only C++ core: the acoustic forward model and the exact
  adjoint-state FWI gradient (2D and 3D), mirroring the Python reference.
- `pybind11` Python bindings, verified to machine precision against the Python
  reference and benchmarked (several times faster).
- MATLAB MEX gateway sharing the same core, with an adjoint gradient check.
- Native C++ example built with CMake.

- Transducer transmit model (`ringfwi.transducer`) closing the transmit loop:
  the FPGA pulser RTL excitation is turned into an acoustic wavelet and
  propagated in the forward model (verified to peak at the intended frequency).
- OpenMP-parallelised core hot loops, and the C++ backend wired into the FWI
  optimiser (`invert(..., backend=uap)`): about 9x faster than pure Python end
  to end on a 49^3 grid, with an identical result.
- GPU backend (`uap_gpu`, CuPy, fused CUDA stencil kernels, FP32): a drop-in
  `backend` that is 1.7x faster than the OpenMP C++ core at 49^3 and 3x at 65^3,
  widening with grid size, and agrees with the FP64 reference to about 1e-6.

Still open in this area: a CUDA path inside the C++ core (so C++/MATLAB get GPU
too), and extending the core with dataset I/O and DAS/TFM.

## Phase 3 — Hardware design (design now, fabricate later)

- Electronics: a 16-channel (scalable) acquisition front-end in KiCad. HV
  pulser, transmit/receive switch, integrated analog front-end, and a Zynq
  SoC for sequencing, capture, and host streaming. See
  `hardware/electronics/`.
- Mechanical: a parametric ring fixture and immersion tank generated from code
  (STEP and STL output), with a central rotatable sample holder. See
  `hardware/mechanical/`.

Deliverable: a manufacturable board package (Gerbers, BOM, pick-and-place) and
printable or machinable mechanical parts.

## Phase 4 — Firmware (RTL in simulation now; bring-up needs hardware)

- FPGA logic developed and verified in simulation ahead of the board, against
  the OpenUSCT software. Done: the receive **capture datapath** (`rx_capture`),
  co-simulated against OpenUSCT acquisitions and reproducing them bit-exactly
  across acquisition delays (each captured frame maps to a UARP Frame). Plus an
  auxiliary delay-and-sum beamformer verified against an integer golden model.
- Still to do: transmit sequencing and focal-law generation, per-channel
  capture and streaming, AXI wrappers and a Zynq block design, and synthesis /
  timing closure in Vivado.
- HAL hardware backend so `acquire()` targets the real board.
- Bring-up and wet testing; close the loop so simulated and acquired data run
  through the identical processing pipeline.

## Phase 5 — Research extensions

- Absorbing boundaries (PML) with a matching adjoint.
- Elastic and anisotropic solvers, enabling crystal-orientation-fabric
  inversion (connecting to the ice-core time-of-flight work, IEEE IUS 2026).
- Multiscale frequency continuation and source estimation.
- Full-resolution 3D (the 3D solver exists; scaling to fine grids needs the
  memory management and speed of the C++/GPU core in Phase 2).
- GPU and FPGA acceleration of the forward and adjoint solves.

## Sequencing logic

Phases 1 and 2 make the platform genuinely useful with no hardware at all, which
is what lets research groups adopt it early. Phase 3 can proceed in parallel
because the hardware targets the same data format the software already consumes.
