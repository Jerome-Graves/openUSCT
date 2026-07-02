# Dataset schema (HDF5)

The `Dataset` is the platform's shared contract. A simulated acquisition and a
real acquisition both produce a file in this layout, and Python, MATLAB, and C++
all read and write it. HDF5 is used because it is language-neutral,
self-describing, and handles large multidimensional arrays; MATLAB and the major
C++ libraries read it natively.

The reference reader/writer is `simulation/ringfwi/dataset.py`. This document is
the language-independent specification any other implementation must match.

## Layout

```
/                                   (root)
  @format_version   string          e.g. "1.0"
  @description      string
  @created_by       string

  /geometry
    @n_elements     int
    @radius_m       float            ring radius, metres
    @centre_freq_hz float
    @array_type     string           "ring" (2D) or "cylinder" (3D)
    element_pos     float[N, dim]     element positions, metres, origin at centre
                                      dim = 2 (planar ring) or 3 (cylinder)

  /acquisition
    @sample_rate_hz    float
    @tx_centre_freq_hz float
    @nominal_speed_m_s float          assumed background speed for imaging
    data               float32[n_tx, n_samples, n_rx]   full-matrix capture
    tx_wavelet         float[n_samples]
    tx_elements        int[n_tx]        element index that produced each transmit
                                        (defaults to 0..n_tx-1 if absent)

  /ground_truth        (optional; simulation only)
    @h_m            float             grid spacing, metres
    c               float[ny, nx]     true sound-speed field (2D), m/s
                    float[nz, ny, nx]     or 3D for a cylinder acquisition
```

## Conventions

- **Units:** SI throughout (metres, seconds, hertz, m/s).
- **FMC indexing:** `data[i, :, j]` is the trace received on element `j` when
  element `i` transmitted.
- **Geometry origin:** the array centre, so a dataset images identically
  regardless of how it was generated.
- **Storage:** channel data is stored as float32 (with gzip) to keep files
  small; readers should not assume more than single precision on disk.
- **Optional groups:** `ground_truth` is present only for simulated data and
  must be ignored by hardware and imaging code.

## Versioning

`format_version` is bumped on any breaking layout change. Readers should check it
and refuse unknown major versions rather than guess.
