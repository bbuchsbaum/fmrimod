"""Compatibility re-exports for dataset metadata accessors."""

from fmrimod.dataset.dataset_methods import (
    all_timepoints,
    blockids,
    blocklens,
    get_run_duration,
    get_run_lengths,
    get_total_duration,
    get_TR,
    is_sampling_frame,
    n_runs,
    n_timepoints,
    samples,
    subject_ids,
)

__all__ = [
    "get_TR",
    "get_run_lengths",
    "get_total_duration",
    "get_run_duration",
    "n_runs",
    "n_timepoints",
    "blocklens",
    "blockids",
    "samples",
    "all_timepoints",
    "subject_ids",
    "is_sampling_frame",
]
