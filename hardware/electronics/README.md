# Electronics

A custom multi-channel ultrasound acquisition front-end: it fires the array,
captures the raw echoes on every channel, digitises them, and streams the
full-matrix-capture dataset to the host.

## Signal chain (per channel)

```
        transmit                         receive
  ┌──────────────┐   ┌───────┐   ┌──────────────────────┐   ┌──────────┐
  │ HV pulser    ├──►│  T/R  ├──►│ AFE                   ├──►│ SoC/FPGA │──► host
  │ (bipolar HV) │   │ switch│   │ LNA · TGC · ADC       │   │ capture  │   (GbE)
  └──────┬───────┘   └───┬───┘   └──────────────────────┘   └────┬─────┘
         │               │                                       │
     transducer ◄────────┘                                  sequencing,
       element                                              beamform-ready
```

- **HV pulser** produces the bipolar high-voltage transmit pulse. Candidate
  parts: STMicro STHV748 (integrated 4-channel pulser) or a discrete
  Microchip MD1210 gate driver with a TC6320 HV MOSFET pair.
- **Transmit/receive (T/R) switch** protects the sensitive receive input from
  the high-voltage transmit pulse. Candidate: MD0100 or an integrated expander.
- **Analog front-end (AFE)** does low-noise amplification, time-gain
  compensation, and analog-to-digital conversion in one device. Candidate:
  Texas Instruments AFE5832LP (32 channels, integrated LNA, variable-gain
  amplifier, and ADC), which covers a 16- or 32-channel system in a single part.
- **SoC / FPGA** sequences the transmit events, captures the digitised channels,
  buffers a full FMC frame, and streams it to the host. Candidate: Xilinx Zynq
  7020 or Zynq UltraScale+ (ARM cores plus programmable logic, with a hardened
  Gigabit Ethernet MAC in the processing system). This is also where later
  on-board beamforming would live.

## Target specification (first revision)

| Parameter | Value |
|---|---|
| Channels | 16 (scalable to 32 on one AFE, more by stacking) |
| Centre frequency | 1 to 5 MHz (NDT / immersion range) |
| Sample rate | 40 to 65 MSPS per channel |
| Transmit | bipolar, programmable per-element delays |
| Acquisition | full-matrix capture (each element Tx, all Rx) |
| Host link | Gigabit Ethernet (Zynq PS) |
| Coupling | immersion (water bath) |

## Design notes

- **Power and safety:** the pulser needs a bipolar HV rail (roughly plus/minus
  50 to 100 V). HV routing, clearance, and interlocks are treated as a
  first-class design constraint, not an afterthought.
- **Clocking and timing:** a shared low-jitter sample clock across all AFE
  channels is what makes coherent beamforming and FWI possible; timing
  distribution is part of the schematic, not left to chance.
- **Scalability:** the 16-channel first revision is deliberately modest so it is
  actually buildable and affordable, while the architecture (one AFE, one SoC)
  scales to a denser array later.

## What lives here

- `openuap-fe.kicad_pro` and friends: the KiCad project (schematic, PCB,
  Gerbers, BOM, pick-and-place), to be built with the KiCad tooling.
- A block diagram and a bill of materials with sourcing.

Fabrication (board order, assembly, transducer wiring, wet test) is lab work for
the maintainer; this directory provides the complete manufacturable package.
