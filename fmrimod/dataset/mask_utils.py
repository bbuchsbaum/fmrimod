"""Canonical mask conversion helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mask_to_logical(mask: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Convert a mask representation to a flat boolean vector."""
    return np.asarray(mask, dtype=np.bool_).ravel()


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
    return np.asarray(mask_vec, dtype=np.bool_).reshape(dims)
