"""Build the libuap Python extension (_uap).

    python setup.py build_ext --inplace

Requires a C++ compiler (MSVC, GCC, or Clang) and pybind11. The C++ core header
lives in ../core and is shared with the MATLAB MEX binding.
"""

import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

core_include = os.path.join(os.path.dirname(__file__), "..", "core")

# OpenMP: MSVC uses /openmp, GCC/Clang use -fopenmp (compile and link).
if sys.platform == "win32":
    omp_compile, omp_link = ["/openmp"], []
else:
    omp_compile, omp_link = ["-fopenmp"], ["-fopenmp"]

ext = Pybind11Extension(
    "_uap",
    ["_uap.cpp"],
    include_dirs=[core_include],
    cxx_std=14,
    extra_compile_args=omp_compile,
    extra_link_args=omp_link,
)

setup(
    name="uap",
    version="0.1.0",
    description="libuap: C++ performance core for OpenUSCT",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
