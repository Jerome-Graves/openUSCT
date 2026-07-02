# Architecture

OpenUSCT is three pillars (hardware, software, simulation) bound together by two
shared contracts: a common **data model** and a common **processing pipeline**.
Get those two right and the pillars compose instead of diverging.

## 1. The shared data model

Everything in the platform speaks in terms of three objects.

**ArrayGeometry** — the transducer element positions and orientations, the ring
radius, element count, centre frequency, and pitch. Identical whether the array
is physical or simulated.

**Dataset** — a full-matrix-capture (FMC) acquisition: for an N-element array, an
`N x N x T` block of raw channel time series (element i transmits, element j
records), plus the sample rate, the transmit waveform, and a reference to the
`ArrayGeometry`. This is the raw product of either the hardware or the simulator.

**Image / Model** — the reconstruction output: a beamformed image, or a
quantitative sound-speed or stiffness map from FWI.

These are serialised to a single portable **HDF5** file (`software/schema/`
defines the layout) so Python, C++, and MATLAB all read and write the same
files. HDF5 is chosen because it is language-neutral, self-describing, handles
large multidimensional arrays, and is already standard in both scientific
Python and MATLAB.

## 2. The shared processing pipeline

```
acquisition  ->  Dataset (HDF5)  ->  processing  ->  Image / Model
 (hw or sim)                          (core + plugins)
```

Because acquisition and simulation both emit a `Dataset`, every processing stage
downstream is source-agnostic. The stages are:

- **Pre-processing / DSP:** bandpass filtering, TGC, Hilbert / analytic signal,
  apodisation.
- **Imaging:** delay-and-sum, total focusing method (TFM), plane-wave
  compounding.
- **Inversion:** adjoint-state full waveform inversion (the working simulation
  core), travel-time tomography for starting models.
- **User plugins:** any algorithm implementing the `Algorithm` interface.

## 3. Language-agnostic software

The requirement is that a user can work in Python, C++, or MATLAB. The way to
deliver that without maintaining three separate implementations:

```
        ┌───────────────────────────┐
        │   libuap  (C++ core)       │   performance-critical:
        │  DSP · beamform · FWI · HAL│   solvers, imaging, hardware I/O
        └───────┬─────────┬──────────┘
    pybind11    │         │   MEX
        ┌───────▼──┐   ┌──▼───────┐
        │  Python  │   │  MATLAB  │      + native C++ users link directly
        └──────────┘   └──────────┘
```

- **One C++ core (`libuap`)** holds the heavy numerics and the hardware
  abstraction layer.
- **Thin bindings** expose it: `pybind11` for Python, a MEX gateway for MATLAB.
  Native C++ code links the library directly.
- **The HDF5 format** is the interop fallback: even without bindings, any
  language can read and write datasets and exchange results.
- The current **Python `ringfwi`** package (in `simulation/`) is the reference
  implementation. It defines the algorithms clearly and readably; the C++ core
  mirrors it for speed. Keeping a readable reference next to the fast core is
  how the project stays verifiable.

### Plugin interface

A custom algorithm implements a small contract: take a `Dataset` (and optional
parameters), return an `Image` or `Model`. Plugins can be written in any of the
three languages and are discovered by name, so a researcher can benchmark a new
beamformer or DSP chain against the built-ins on identical data.

## 4. Hardware abstraction layer (HAL)

The hardware and the simulator present the **same acquisition API**:

```
acquire(geometry, sequence) -> Dataset
```

For the PCB, `acquire` drives the pulser sequence and reads back digitised
channels over the host link. For the simulator, `acquire` runs the wave solver.
Application code targets the HAL, so the same experiment script runs against a
board or against a simulation by swapping one backend.

## 5. Conventions

- **Units:** SI throughout (metres, seconds, hertz, m/s). Model parameter for the
  solvers is squared slowness `m = 1/c^2`.
- **Dimension:** the solver, geometry, and dataset are dimension-general. A
  planar ring gives a 2D field indexed `[iy, ix]`; a cylindrical array (rings
  stacked along the axis) gives a 3D field indexed `[iz, iy, ix]`. Element
  positions are stored as `(N, dim)` with `dim` 2 or 3.
- **Axes:** row = y, column = x (and slice = z in 3D); geometry in metres with
  the domain centre as origin for physical coordinates.
- **Time:** datasets store an explicit sample rate; nothing assumes a fixed one.

See [../simulation/docs/theory.md](../simulation/docs/theory.md) for the wave
physics and the adjoint-state gradient that the FWI core is built on.
