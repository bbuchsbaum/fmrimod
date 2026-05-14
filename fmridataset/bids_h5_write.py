"""Compatibility re-exports for BIDS-HDF5 study writers."""

from fmrimod.dataset.bids_h5_write import *  # noqa: F403
from fmrimod.dataset.bids_h5_write import __all__ as _all

__all__ = list(_all)
