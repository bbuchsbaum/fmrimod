"""Compatibility re-exports for dataset constructors."""

from fmrimod.dataset.constructors import fmri_dataset, matrix_dataset

__all__ = ["matrix_dataset", "fmri_dataset"]
