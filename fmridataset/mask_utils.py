"""Compatibility re-exports for mask conversion helpers."""

from fmrimod.dataset.mask_utils import mask_to_logical, mask_to_volume

__all__ = ["mask_to_logical", "mask_to_volume"]
