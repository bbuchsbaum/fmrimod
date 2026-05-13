"""Canonical dataset constructors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.sampling import SamplingFrame

from .adapters import BackendAdapter
from .backend_constructors import matrix_backend
from .fmri_dataset import FmriDataset
from .protocols import DatasetProtocol


def fmri_dataset(
    data_source: DatasetProtocol,
    event_table: pd.DataFrame | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
) -> FmriDataset:
    """Construct the canonical dataset container from a data source."""
    return FmriDataset(data_source, event_table=event_table, censor=censor)


def matrix_dataset(
    data: Any,
    tr: float | list[float] | None = None,
    run_length: int | list[int] | None = None,
    *,
    event_table: pd.DataFrame | None = None,
    mask: NDArray[np.bool_] | None = None,
    TR: float | list[float] | None = None,
) -> FmriDataset:
    """Construct an in-memory canonical dataset from matrix data.

    Parameters
    ----------
    data
        Either a single ``time x voxels`` matrix or a list of per-run matrices.
    tr
        Repetition time, or per-run repetition times.
    TR
        Compatibility spelling for ``tr``. If both are supplied, they must
        agree.
    run_length
        Run length or lengths used to split a single matrix input.
    event_table
        Optional event table attached to the resulting dataset.
    mask
        Optional flat or spatial boolean mask.
    """
    resolved_tr = _resolve_tr(tr, TR)

    if isinstance(data, list):
        runs = [np.asarray(x, dtype=np.float64) for x in data]
        if not runs:
            raise ValueError("matrix_dataset requires at least one run")
        if any(r.ndim != 2 for r in runs):
            raise ValueError("matrix_dataset expects 2-D matrix data")
        n_cols = {int(r.shape[1]) for r in runs}
        if len(n_cols) != 1:
            raise ValueError("all run matrices must have the same number of columns")
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("matrix_dataset expects 2-D matrix data")
        if run_length is None:
            runs = [arr]
        else:
            blocklens = _normalize_run_lengths(run_length, n_rows=int(arr.shape[0]))
            splits = np.cumsum(blocklens)[:-1]
            runs = [
                np.asarray(x, dtype=np.float64) for x in np.split(arr, splits, axis=0)
            ]

    blocklens = [int(r.shape[0]) for r in runs]
    sampling_frame = SamplingFrame(blocklens=blocklens, tr=resolved_tr)
    data_matrix = np.vstack(runs)
    flat_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    backend = matrix_backend(data_matrix, mask=flat_mask)
    adapter = BackendAdapter(backend, sampling_frame)
    return fmri_dataset(adapter, event_table=event_table)


def _resolve_tr(
    tr: float | list[float] | None,
    TR: float | list[float] | None,
) -> float | list[float]:
    """Resolve canonical and compatibility TR spellings."""
    if tr is None and TR is None:
        raise ValueError("matrix_dataset requires tr or TR")
    if tr is not None and TR is not None:
        if not np.array_equal(np.asarray(tr), np.asarray(TR)):
            raise ValueError("tr and TR must agree when both are supplied")
    return TR if tr is None else tr


def _normalize_run_lengths(
    run_length: int | list[int],
    *,
    n_rows: int,
) -> list[int]:
    """Normalize split specification for a single matrix input."""
    if isinstance(run_length, int):
        if run_length <= 0 or (n_rows % run_length) != 0:
            raise ValueError(
                "run_length must divide number of timepoints for matrix input"
            )
        return [run_length] * (n_rows // run_length)

    blocklens = [int(x) for x in run_length]
    if any(x <= 0 for x in blocklens):
        raise ValueError("run_length values must be positive")
    if sum(blocklens) != n_rows:
        raise ValueError("sum(run_length) must equal number of timepoints for matrix input")
    return blocklens
