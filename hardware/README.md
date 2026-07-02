# Hardware

The physical layer of OpenUSCT: the transducer array, the custom acquisition PCB
that drives it, and the fixture and tank that hold everything in a known
geometry.

- **[electronics/](electronics/)** — the custom acquisition front-end: high
  voltage pulser, transmit/receive switch, integrated analog front-end, and a
  Zynq SoC that captures full-matrix-capture data and streams it to the host.
  Designed in KiCad.
- **[mechanical/](mechanical/)** — the ring fixture, immersion tank, and rotary
  sample holder, generated parametrically from code as STEP and STL.

Both are specified so that what they produce (channel data on a known array
geometry) is exactly the `Dataset` the [software](../software/) pillar consumes,
which is the same object the [simulation](../simulation/) pillar produces. The
hardware is a data source behind the common hardware abstraction layer; nothing
downstream needs to know whether a dataset came from a board or from a solver.

Fabrication and assembly are lab work for the maintainer. This pillar provides
the complete, open design package needed to build it.
