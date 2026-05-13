"""Compatibility facade for the former ``fmridataset`` package.

The implementation source of truth is ``fmrimod.dataset``. This package exists
only to keep old imports working during the consolidation.
"""

from __future__ import annotations

from fmrimod import __version__
from fmrimod.dataset import *  # noqa: F403
from fmrimod.dataset import __all__ as _dataset_all
from fmrimod.sampling import SamplingFrame

__all__ = ["__version__", "SamplingFrame", *_dataset_all]
