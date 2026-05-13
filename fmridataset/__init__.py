"""Compatibility facade for the former ``fmridataset`` package.

The implementation source of truth is ``fmrimod.dataset``. This package exists
only to keep old imports working during the consolidation.
"""

from __future__ import annotations

import sys
from types import ModuleType

from fmrimod import __version__
from fmrimod.dataset import *  # noqa: F403
from fmrimod.dataset import __all__ as _dataset_all
from fmrimod.dataset import data_chunks as _canonical_data_chunks
from fmrimod.dataset import fmri_series as _canonical_fmri_series
from fmrimod.sampling import SamplingFrame

__all__ = ["__version__", "SamplingFrame", *_dataset_all]


class _FacadeModule(ModuleType):
    """Keep root function re-exports stable after submodule imports."""

    def __getattribute__(self, name: str):
        if name == "data_chunks":
            return _canonical_data_chunks
        if name == "fmri_series":
            return _canonical_fmri_series
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _FacadeModule
