# Software

One processing stack for OpenUSCT, usable from **Python, C++, or MATLAB**, running
identically on simulated or acquired data.

## Design

The platform keeps one performant C++ core and exposes it through thin bindings,
with a portable HDF5 file format as the universal fallback. See
[../docs/architecture.md](../docs/architecture.md) for the rationale.

```
        core/libuap.hpp  (header-only C++):  forward model + exact adjoint FWI gradient
              │                     │                    │
        pybind11 (Python)     MEX (MATLAB)         native C++
              └─────────── HDF5 Dataset format ───────────┘
```

The C++ core mirrors the readable Python reference in `simulation/ringfwi`
operation for operation, so on identical inputs the two agree to machine
precision. That parity is the verification contract for the whole pillar.

## Layout

```
software/
  core/
    libuap.hpp        header-only C++ core (2D/3D forward + adjoint gradient)
    example_main.cpp  native C++ example / smoke test
    CMakeLists.txt    CMake build for the native example
  python/
    _uap.cpp          pybind11 bindings
    uap.py            drop-in Python front end (same API as ringfwi.fwi)
    setup.py          build the _uap extension
    test_parity.py    C++ vs Python parity + benchmark
  matlab/
    uap_mex.cpp       MEX gateway (shares core/libuap.hpp)
    uap_forward_fmc.m, uap_misfit_and_gradient.m, subs2lin.m   .m wrappers
    uap_gradient_check.m   MEX adjoint gradient check
  schema/
    README.md         the portable HDF5 Dataset specification
```

## Building and verifying

**Python (pybind11).** Requires a C++ compiler and `pybind11`.

```bash
cd python
python setup.py build_ext --inplace
python test_parity.py
```

The C++ core matches the Python reference to about 1e-15 in 2D and 3D and runs
several times faster. After building, use it as a drop-in fast backend:

```python
import uap as fwi          # same forward_fmc / misfit_and_gradient as ringfwi.fwi
```

**MATLAB (MEX).** Requires MATLAB with a configured C++ compiler.

```matlab
mex -R2018a -I../core uap_mex.cpp
uap_gradient_check          % adjoint vs finite-difference check
```

**Native C++ (CMake).**

```bash
cd core
cmake -S . -B build
cmake --build build --config Release
./build/Release/uap_example      # (or build/uap_example on Unix)
```

## The plugin contract

A custom algorithm takes a `Dataset` (plus optional parameters) and returns an
`Image` or a `Model`, discovered by name. Implemented today in the Python
reference (`ringfwi.plugins`); the C++/MATLAB paths use the same names.

## Status

- **HDF5 `Dataset` format** (2D and 3D): done, with round-trip tests.
- **Python reference** (imaging, FWI, plugins, full pipeline): done.
- **C++ core `libuap`** (forward + exact adjoint gradient, 2D and 3D): done,
  verified to machine precision against the Python reference.
- **Python bindings** (pybind11): done and benchmarked (several times faster).
- **MATLAB bindings** (MEX): done, adjoint gradient check passes.
- **Native C++**: builds and runs via CMake.
- **OpenMP**: the core hot loops are parallelised (`/openmp`), giving about 2x
  more on large grids (the stencil solver is memory-bandwidth bound).
- **Optimiser integration**: `ringfwi.fwi.invert(..., backend=uap)` runs the
  whole FWI loop on the C++ core. End to end on a 49^3 grid this is about 9x
  faster than the pure-Python path, with an identical result.
- **GPU backend** (`uap_gpu`, CuPy): the 3D path uses fused CUDA stencil kernels
  and runs in FP32 (consumer GPUs throttle FP64). It agrees with the FP64 CPU
  reference to about 1e-6 and, as another drop-in `backend`, is 1.7x faster than
  the OpenMP C++ core at 49^3 and 3x at 65^3, widening with grid size.

```bash
pip install "cupy-cuda12x[ctk]"     # NVIDIA GPU; no system CUDA toolkit needed
python test_gpu.py
```

```python
import uap_gpu
fwi.invert(..., backend=uap_gpu)     # run the FWI loop on the GPU
```

Next: extend the C++ core with dataset I/O and DAS/TFM, and a CUDA path callable
from C++/MATLAB too. See the [roadmap](../docs/roadmap.md).
