"""Algorithm plugin interface.

The contract is deliberately small: an algorithm is a callable that takes a
:class:`~ringfwi.dataset.Dataset` (plus optional parameters) and returns a
result (an image or a model). Algorithms register by name, so a researcher can
drop in a new beamformer, inversion, or DSP chain and run it on the same data as
the built-ins, on equal footing.

    from ringfwi import plugins

    @plugins.register("my_method", description="my custom imaging")
    def my_method(dataset, **params):
        ...
        return image

    result = plugins.run("my_method", dataset)

This is what makes the platform a place to try current algorithms and DSP,
rather than a fixed pipeline. The same registry is intended to back the C++ and
MATLAB implementations, discovered by the same names.
"""

from __future__ import annotations

_REGISTRY = {}


def register(name, description="", output="image"):
    """Decorator: register ``fn`` as an algorithm callable(dataset, **params)."""
    def deco(fn):
        _REGISTRY[name] = {"fn": fn, "description": description, "output": output}
        return fn
    return deco


def available():
    """Return {name: description} for all registered algorithms."""
    return {k: v["description"] for k, v in _REGISTRY.items()}


def info(name):
    """Return the registry entry for ``name``."""
    return _REGISTRY[name]


def run(name, dataset, **params):
    """Run a registered algorithm on a dataset."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]["fn"](dataset, **params)


# --- built-in algorithms ----------------------------------------------------

@register("tfm", description="Total focusing method (delay-and-sum) envelope image")
def _tfm(dataset, npix=120, half_size=None, envelope=True):
    from .imaging import tfm
    image, _axes = tfm(dataset, npix=npix, half_size=half_size, envelope=envelope)
    return image


@register("fwi", description="Adjoint-state full waveform inversion (sound-speed model)",
          output="model")
def _fwi(dataset, **params):
    from .fwi import fwi_from_dataset
    return fwi_from_dataset(dataset, **params)
