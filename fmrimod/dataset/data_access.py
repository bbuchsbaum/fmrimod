"""Canonical free-functions for dataset data access."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fmri_dataset import FmriDataset


def get_data(
    dataset: FmriDataset,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.floating[Any]]:
    """Extract a data matrix from a dataset."""
    return dataset.get_data(rows=rows, cols=cols)


def get_data_matrix(
    dataset: FmriDataset,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.floating[Any]]:
    """Extract data as a ``timepoints x voxels`` matrix."""
    if hasattr(dataset, "get_data_matrix"):
        return dataset.get_data_matrix(rows=rows, cols=cols)
    return dataset.get_data(rows=rows, cols=cols)


def get_mask(dataset: FmriDataset) -> NDArray[np.bool_]:
    """Extract the dataset mask."""
    return dataset.get_mask()
