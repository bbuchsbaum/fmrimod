"""Constructor helpers for canonical storage backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backends.latent_backend import LatentBackend
from .backends.matrix_backend import MatrixBackend
from .backends.nifti_backend import NiftiBackend


def matrix_backend(
    data_matrix: NDArray[np.float64],
    *,
    mask: NDArray[np.bool_] | None = None,
    spatial_dims: tuple[int, int, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> MatrixBackend:
    """Construct an in-memory matrix backend."""
    return MatrixBackend(
        data_matrix,
        mask=mask,
        spatial_dims=spatial_dims,
        metadata=metadata,
    )


def latent_backend(
    source: str | Path | Sequence[str | Path],
    *,
    preload: bool = False,
) -> LatentBackend:
    """Construct a storage-backed latent decomposition backend."""
    return LatentBackend(source=source, preload=preload)


def nifti_backend(
    source: str | Path | Sequence[str | Path],
    mask_source: str | Path,
    *,
    preload: bool = False,
) -> NiftiBackend:
    """Construct a NIfTI storage backend."""
    return NiftiBackend(source=source, mask_source=mask_source, preload=preload)
