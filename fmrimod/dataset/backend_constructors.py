"""Constructor helpers for canonical storage backends."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backends.matrix_backend import MatrixBackend


def matrix_backend(
    data_matrix: NDArray[np.floating[Any]],
    *,
    mask: NDArray[np.bool_] | None = None,
    spatial_dims: tuple[int, int, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MatrixBackend:
    """Construct an in-memory matrix backend."""
    return MatrixBackend(
        data_matrix,
        mask=mask,
        spatial_dims=spatial_dims,
        metadata=metadata,
    )
