"""Canonical functional accessors for dataset temporal metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmrimod.sampling import SamplingFrame

from .fmri_dataset import FmriDataset


def _frame(obj: FmriDataset | SamplingFrame) -> SamplingFrame:
    if isinstance(obj, SamplingFrame):
        return obj
    if isinstance(obj, FmriDataset):
        return obj.sampling_frame
    raise TypeError("expected a FmriDataset or SamplingFrame")


def is_sampling_frame(obj: object) -> bool:
    """Return whether *obj* is the canonical sampling frame type."""
    return isinstance(obj, SamplingFrame)


def get_TR(obj: FmriDataset | SamplingFrame) -> float:  # noqa: N802
    """Return the first repetition time in seconds."""
    return _frame(obj).TR


def get_run_lengths(obj: FmriDataset | SamplingFrame) -> tuple[int, ...]:
    """Return per-run lengths in timepoints."""
    return tuple(int(v) for v in _frame(obj).blocklens)


def blocklens(obj: FmriDataset | SamplingFrame) -> tuple[int, ...]:
    """Alias for :func:`get_run_lengths`."""
    return get_run_lengths(obj)


def n_runs(obj: FmriDataset | SamplingFrame) -> int:
    """Return the number of runs."""
    return int(_frame(obj).n_blocks)


def n_timepoints(obj: FmriDataset | SamplingFrame) -> int:
    """Return the total number of timepoints."""
    if isinstance(obj, FmriDataset):
        return obj.n_timepoints
    return int(obj.n_scans)


def blockids(
    obj: FmriDataset | SamplingFrame,
    *,
    one_based: bool = False,
) -> NDArray[np.int32]:
    """Return run ids for each timepoint.

    The canonical default is 0-based. R-style 1-based ids require
    ``one_based=True``.
    """
    return _frame(obj).run_ids(one_based=one_based)


def samples(obj: FmriDataset | SamplingFrame) -> NDArray[np.float64]:
    """Return acquisition/sample times in seconds."""
    return _frame(obj).samples


def get_total_duration(obj: FmriDataset | SamplingFrame) -> float:
    """Return total acquisition duration in seconds."""
    frame = _frame(obj)
    return float(np.sum(frame.blocklens * frame.tr))


def get_run_duration(obj: FmriDataset | SamplingFrame) -> NDArray[np.float64]:
    """Return per-run acquisition durations in seconds."""
    frame = _frame(obj)
    return np.asarray(frame.blocklens * frame.tr, dtype=np.float64)


def all_timepoints(dataset: FmriDataset) -> NDArray[np.intp]:
    """Return all valid 0-based timepoint indices for *dataset*."""
    return np.arange(dataset.n_timepoints, dtype=np.intp)


def subject_ids(dataset: Any) -> list[Any]:
    """Return subject identifiers from a study-like dataset."""
    ids = getattr(dataset, "subject_ids", None)
    if ids is None:
        raise TypeError("expected an object with subject_ids")
    return list(ids)
