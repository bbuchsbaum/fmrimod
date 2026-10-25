"""Canonical mask conversion helpers.

Every mask that enters ``fmrimod.dataset`` (arrays, NIfTI or NeuroVol images,
default masks from a first volume) is binarised by one rule, R's
``which(as.logical(x))`` (fmridataset ``read-bids-bold.R``,
``nifti-array-source.R``): a voxel is in the mask when its value is non-zero,
negatives included, and NaN is out.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_mask(values: ArrayLike) -> NDArray[np.bool_]:
    """Binarise mask values, keeping their shape.

    Booleans pass through; numbers are in when non-zero and not NaN.
    """
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.number):
        out = arr != 0
        if np.issubdtype(arr.dtype, np.inexact):
            out &= ~np.isnan(arr)
        return np.asarray(out, dtype=np.bool_)
    return np.asarray(arr, dtype=np.bool_)


def mask_to_logical(mask: ArrayLike) -> NDArray[np.bool_]:
    """Convert a mask representation to a flat boolean vector (see :func:`as_mask`)."""
    return as_mask(mask).ravel()


def mask_to_volume(
    mask_vec: NDArray[np.bool_],
    dims: tuple[int, int, int],
) -> NDArray[np.bool_]:
    """Reshape a flat boolean mask into a 3-D boolean volume."""
    expected = int(np.prod(dims))
    if mask_vec.size != expected:
        raise ValueError(
            f"Mask length ({mask_vec.size}) doesn't match spatial "
            f"dimensions {dims} (product={expected})"
        )
    return as_mask(mask_vec).reshape(dims)
