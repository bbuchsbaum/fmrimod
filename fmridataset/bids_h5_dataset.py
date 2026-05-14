"""Compatibility re-exports for BIDS-HDF5 study readers."""

from fmrimod.dataset.bids_h5 import *  # noqa: F403
from fmrimod.dataset.bids_h5 import __all__ as _ALL

__all__ = list(_ALL)
