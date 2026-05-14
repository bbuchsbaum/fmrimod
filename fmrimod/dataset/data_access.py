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


def get_run_data(dataset: FmriDataset, run: int) -> NDArray[np.floating[Any]]:
    """Extract one run from a dataset through the explicit run-access seam."""
    run_method = getattr(dataset, "get_run_data", None)
    if callable(run_method):
        return run_method(int(run))
    return dataset.get_data(int(run))


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
