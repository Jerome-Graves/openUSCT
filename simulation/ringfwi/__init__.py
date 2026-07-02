"""OpenUSCT: full waveform inversion for ring-array ultrasound tomography.

A compact, dependency-light research framework for reconstructing the sound
speed of a cylindrical specimen from full-matrix-capture data recorded on a
ring transducer array. The forward model is a 2D acoustic finite-difference
solver; the inverse solver is adjoint-state full waveform inversion.

(``ringfwi`` is the import package name; OpenUSCT is the project.)
"""

from . import (acquire, anisotropy, attenuation, dataset, elastic, fwi,
               geometry, imaging, phantom, plugins, render3d, solver, sources,
               transducer, uarp_format)
from .dataset import ArrayGeometry, Dataset

__all__ = [
    "acquire",
    "anisotropy",
    "attenuation",
    "dataset",
    "elastic",
    "fwi",
    "geometry",
    "imaging",
    "phantom",
    "plugins",
    "render3d",
    "solver",
    "sources",
    "transducer",
    "uarp_format",
    "ArrayGeometry",
    "Dataset",
]
__version__ = "0.1.0"
