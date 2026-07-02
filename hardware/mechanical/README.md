# Mechanical

The fixture that holds the transducer array around the specimen, and the
immersion tank that couples sound into it.

## What it has to do

- Hold N transducer elements on a precise ring of known radius, each aimed at the
  ring centre, so the geometry matches the `ArrayGeometry` the software assumes.
- Present the specimen at the centre, coaxial with the ring.
- Provide a water bath for immersion coupling.
- Allow a **rotatable central sample holder**. Rotating the specimen between
  acquisitions synthesises a denser angular aperture from a modest element count,
  and it directly supports azimuthal scanning of cylindrical specimens such as
  ice cores (the same acquisition idea as the time-of-flight fabric work).

## Parametric, code-generated CAD

Rather than depend on a proprietary CAD package, the parts are generated
parametrically from Python using a modelling kernel (CadQuery or build123d) and
exported to **STEP** (for machining or import into any CAD) and **STL** (for 3D
printing). Parameters: element count, ring radius, element diameter, tank size,
wall thickness.

Benefits: the fixture geometry is defined by the same numbers as the software
`ArrayGeometry`, kept in one place; anyone can regenerate the parts for a
different array without a CAD licence; and the output formats are open.

## What lives here

- `gen_fixture.py`: parametric generator producing the ring holder, tank, and
  sample mount, exporting STEP and STL.
- Generated meshes and drawings.
- Assembly notes and a suggested bill of materials (bearings for the rotary
  stage, seals, fasteners).

## Status

Specification stage. The generator script and exported parts are the next
mechanical deliverable, sized to match the 16-element first-revision array in
`../electronics`.
