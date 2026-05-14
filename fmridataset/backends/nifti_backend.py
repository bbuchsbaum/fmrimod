"""Compatibility re-export for the NIfTI storage backend."""

from fmrimod.dataset.backends.nifti_backend import NiftiBackend, nifti_backend

__all__ = ["NiftiBackend", "nifti_backend"]
