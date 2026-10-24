"""Canonical dataset constructors.

The public entry is :func:`fmri_dataset`. It accepts a polymorphic image
source and returns a :class:`FmriDataset` ready to feed
:func:`fmrimod.glm.fmri_lm`.

Supported inputs for ``img``:

- :class:`neuroim.NeuroVec` (or sequence, one per run) — canonical NIfTI path.
- ``nibabel.Nifti1Image`` (or sequence) — compatibility shim, wrapped via
  :class:`~fmrimod.dataset.adapters.NibabelAdapter`.
- ``str`` / :class:`pathlib.Path` (or sequence) — loaded via
  :func:`neuroim.read_vec`.
- 4-D ``ndarray`` of shape ``(x, y, z, t)`` — wrapped as a single-run NeuroVec.
- 2-D ``ndarray`` of shape ``(time, voxels)`` — matrix path.
- Anything satisfying :class:`DatasetProtocol` is accepted as-is.

The legacy single-positional form ``fmri_dataset(data_source)`` continues to
work for callers that already hold an adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.sampling import SamplingFrame, same_seconds

from .adapters import BackendAdapter
from .adapters.neuroim_adapter import _is_neurovec
from .backend_constructors import matrix_backend
from .fmri_dataset import FmriDataset
from .protocols import DatasetProtocol


def fmri_dataset(
    img: object = None,
    *,
    mask: object = None,
    tr: float | Sequence[float] | None = None,
    events: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
    start_time: float | Sequence[float] | None = None,
    slice_timing_offset: float | Sequence[float] | None = None,
    run_length: int | Sequence[int] | None = None,
    confounds: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    # legacy spellings:
    data_source: DatasetProtocol | None = None,
    event_table: pd.DataFrame | None = None,
) -> FmriDataset:
    """Construct the canonical :class:`FmriDataset` from an image source.

    Parameters
    ----------
    img
        Image source. See module docstring for accepted types.
    mask
        Optional mask. Accepts a NeuroVol/LogicalNeuroVol, 3-D boolean ndarray,
        nibabel image, or path to a NIfTI volume. Defaults to non-zero voxels
        of the first volume.
    tr
        Repetition time in seconds. Required for image inputs.
    events
        Optional event table: one DataFrame with a ``run`` column (1-based)
        for multi-run data, or a sequence of per-run DataFrames (BIDS
        ``*_events.tsv`` style, onsets relative to each run's start), which
        is stacked into one table with ``run`` filled in.
    censor
        Optional boolean censor vector(s).
    start_time, slice_timing_offset
        Slice-timing offset in seconds; the realised sampling grid is
        ``start_time + k * TR`` for ``k`` in ``0..n_scans-1``. Default
        is ``TR/2`` (BOLD-midpoint convention — the value at sample
        ``k`` represents the BOLD signal at the middle of the
        ``k``-th TR window). Pass ``slice_timing_offset=0.0`` for
        frame-start sampling (Nilearn / SPM-MAT / FitLins convention).
        The two names are interchangeable; if both are supplied they
        must agree.
    run_length
        Split a single 2-D ``(time, voxels)`` matrix into multiple runs.
        Mirrors :func:`matrix_dataset`'s ``run_length``. Only valid for
        2-D ndarray input; raises ``ValueError`` for any other ``img``
        type, including sequences (which already encode runs explicitly)
        and 4-D ndarrays.
    confounds
        Optional nuisance tables, one per run (or one table covering all
        runs back to back). They are stored on the dataset; a spec term
        ``confounds("trans_x", ...)`` without ``source=`` picks its columns
        from them at fit time.
    data_source
        Legacy alias accepting a :class:`DatasetProtocol`-compatible object.
    event_table
        Legacy alias for ``events``.

    Examples
    --------
    Two runs from NIfTI files with per-run BIDS event tables::

        ds = fmri_dataset(
            ["run-1_bold.nii.gz", "run-2_bold.nii.gz"],
            mask="brain_mask.nii.gz",
            tr=2.0,
            events=[events_run1, events_run2],
        )

    A ``(time, voxels)`` matrix split into two runs:

    >>> import numpy as np, pandas as pd
    >>> events = pd.DataFrame({"onset": [4.0, 4.0], "duration": 1.0,
    ...                        "trial_type": ["a", "b"], "run": [1, 2]})
    >>> ds = fmri_dataset(np.zeros((40, 3)), tr=2.0, run_length=[20, 20],
    ...                   events=events)
    >>> ds.n_runs
    2
    """
    n_event_tables = (
        None
        if events is None or isinstance(events, pd.DataFrame)
        else len(list(events))
    )
    resolved_events = _resolve_events(events, event_table)
    resolved_start_time = _resolve_start_time(start_time, slice_timing_offset)

    if data_source is not None:
        if img is not None:
            raise ValueError("Pass `img` or `data_source`, not both")
        if run_length is not None:
            raise ValueError("`run_length` is not valid with `data_source`")
        dataset = FmriDataset(
            data_source,
            event_table=resolved_events,
            censor=censor,
            confounds=_as_frame_list(confounds),
        )
        _check_events_cover_runs(dataset, n_event_tables)
        return dataset

    if img is None:
        raise ValueError("fmri_dataset requires `img` (or legacy `data_source`)")

    # Per-run ``tr``/``start_time`` pass through to the adapter's
    # SamplingFrame; ``None`` lets it apply ``TR/2`` per run (R 99d19fe, #94).
    adapter = _build_adapter(
        img,
        mask=mask,
        tr=tr,
        start_time=resolved_start_time,
        run_length=run_length,
    )
    dataset = FmriDataset(
        adapter,
        event_table=resolved_events,
        censor=censor,
        confounds=_as_frame_list(confounds),
    )
    _check_events_cover_runs(dataset, n_event_tables)
    return dataset


def _check_events_cover_runs(dataset: FmriDataset, n_event_tables: int | None) -> None:
    """A per-run list of event tables must have one table per run."""
    n_runs = int(dataset.n_runs)
    if n_event_tables is not None and n_event_tables != n_runs:
        raise ValueError(
            f"{n_event_tables} events table(s) for {n_runs} runs; pass one "
            "table per run, or one table with a `run` column"
        )


def require_run_labels(events: pd.DataFrame, n_runs: int) -> str | None:
    """Return the run-label column for a multi-run fit, or raise.

    With several runs, every event needs a run label (a 1-based ``run`` or
    ``block`` column); without one all events would land in run 1 and the
    other runs would be fitted as pure baseline. Returns ``None`` for a
    single run without a label.
    """
    label = next((c for c in ("run", "block") if c in events.columns), None)
    if label is None:
        if n_runs > 1:
            raise ValueError(
                f"the dataset has {n_runs} runs but the events table has no "
                "`run` column; add one (1-based) or build the dataset with "
                "events=[table_run1, table_run2, ...]"
            )
        return None
    if n_runs > 1:
        runs = pd.to_numeric(events[label], errors="coerce")
        bad = runs.isna() | ~runs.between(1, n_runs)
        if bad.any():
            raise ValueError(
                f"events `{label}` values must be run numbers 1..{n_runs}; "
                f"got {sorted(set(events[label][bad].astype(str)))[:5]}"
            )
    return label


def _as_frame_list(
    confounds: pd.DataFrame | Sequence[pd.DataFrame] | None,
) -> pd.DataFrame | list[pd.DataFrame] | None:
    if confounds is None or isinstance(confounds, pd.DataFrame):
        return confounds
    return list(confounds)


def coerce_dataset(
    source: object,
    *,
    tr: object,
    events: object,
    mask: object,
    slice_timing_offset: object,
) -> object:
    """Return ``source`` if it is a dataset, else build one from an image.

    Shared by the entry points that accept either an :class:`FmriDataset` or
    an image source plus ``tr=``/``events=``/``mask=`` shorthand
    (:func:`fmrimod.fmri_lm`, :func:`fmrimod.estimate_single_trial_from_dataset`).
    """
    shorthand = {
        "tr": tr,
        "events": events,
        "mask": mask,
        "slice_timing_offset": slice_timing_offset,
    }
    given = {key: value for key, value in shorthand.items() if value is not None}
    is_dataset = source is None or (
        hasattr(source, "get_sampling_frame") and hasattr(source, "event_table")
    )
    if is_dataset:
        if given:
            raise ValueError(
                f"{sorted(given)} build a dataset from an image and "
                "cannot be combined with a prebuilt dataset; pass them to "
                "fmri_dataset(...) instead."
            )
        return source
    return fmri_dataset(source, **given)  # type: ignore[arg-type]


def matrix_dataset(
    data: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    tr: float | list[float] | None = None,
    run_length: int | list[int] | None = None,
    *,
    event_table: pd.DataFrame | None = None,
    mask: NDArray[np.bool_] | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
    start_time: float | Sequence[float] | None = None,
    slice_timing_offset: float | Sequence[float] | None = None,
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
    censor
        Optional per-run censor mask, or a single flat censor vector that
        will be split along ``run_length``. ``True`` marks volumes to
        exclude during fitting. Consumed by ``fmri_lm`` strategies that
        honor censoring (currently the runwise and concat engines).
    start_time, slice_timing_offset
        Slice-timing offset in seconds (per-run if provided as a sequence,
        or scalar broadcast across all runs). The realised sampling grid
        is ``start_time + k * TR`` for ``k`` in ``0..n_scans-1``. Default
        is ``TR/2`` (midpoint convention — the value at sample ``k``
        represents the BOLD signal at the middle of the ``k``-th TR
        window). Pass ``slice_timing_offset=0.0`` for frame-start
        sampling (Nilearn / SPM-MAT / FitLins convention). The two names
        are interchangeable; if both are supplied they must agree.
    """
    resolved_tr = _resolve_tr(tr, TR)
    resolved_start_time = _resolve_start_time(start_time, slice_timing_offset)


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
    sampling_frame = SamplingFrame(
        blocklens=blocklens, tr=resolved_tr, start_time=resolved_start_time
    )
    data_matrix = np.vstack(runs)
    flat_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    backend = matrix_backend(data_matrix, mask=flat_mask)
    adapter = BackendAdapter(backend, sampling_frame)
    # Split a flat censor across runs to match the multi-block sframe;
    # the per-run-list form is passed through to FmriDataset as-is.
    resolved_censor: Any = censor
    if isinstance(censor, np.ndarray) and censor.ndim == 1 and len(runs) > 1:
        run_lengths = [int(r.shape[0]) for r in runs]
        splits = np.cumsum(run_lengths)[:-1]
        resolved_censor = list(np.split(np.asarray(censor, dtype=bool), splits))
    return FmriDataset(
        adapter, event_table=event_table, censor=resolved_censor
    )


# -- Internal helpers ----------------------------------------------------------


def _resolve_events(
    events: pd.DataFrame | Sequence[pd.DataFrame] | None,
    event_table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if events is not None and event_table is not None and event_table is not events:
        raise ValueError("Pass `events` or `event_table`, not both")
    resolved = events if events is not None else event_table
    if resolved is None or isinstance(resolved, pd.DataFrame):
        return resolved
    return _stack_run_events(resolved)


def _stack_run_events(per_run: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-run event tables into one table with a 1-based ``run``."""
    frames = list(per_run)
    if not frames or not all(isinstance(f, pd.DataFrame) for f in frames):
        raise TypeError(
            "events must be a DataFrame or a non-empty sequence of per-run "
            "DataFrames"
        )
    stacked = []
    for run, frame in enumerate(frames, start=1):
        if "run" in frame.columns and not (frame["run"] == run).all():
            raise ValueError(
                f"per-run events table {run - 1} has a `run` column that is "
                f"not {run}; pass a single stacked table instead"
            )
        stacked.append(frame.assign(run=run))
    return pd.concat(stacked, ignore_index=True)


def _is_dataset_protocol(obj: object) -> bool:
    """Duck-type check; avoid runtime_checkable Protocol cost for hot path."""
    return (
        hasattr(obj, "get_data")
        and hasattr(obj, "get_mask")
        and hasattr(obj, "get_sampling_frame")
        and hasattr(obj, "n_runs")
    )


def _is_nifti1image(obj: object) -> bool:
    try:
        import nibabel
    except ImportError:
        return False
    return isinstance(obj, cast(Any, nibabel).Nifti1Image)


def _build_adapter(
    img: object,
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
    run_length: int | Sequence[int] | None = None,
) -> DatasetProtocol:
    # `run_length` only has a meaning for a single 2-D matrix input.
    # Sequences (NeuroVec/Nifti/path lists) already encode runs explicitly,
    # and 4-D ndarrays carry their own time dimension semantics.
    if run_length is not None and not (
        isinstance(img, np.ndarray) and img.ndim == 2
    ):
        raise ValueError(
            "`run_length` is only valid for 2-D ndarray input; "
            "for image sequences, pass a list of runs instead"
        )

    # DatasetProtocol-compatible: use as-is.
    if _is_dataset_protocol(img):
        return cast(DatasetProtocol, img)

    # Sequence dispatch
    if isinstance(img, (list, tuple)) and img:
        first = img[0]
        if _is_neurovec(first) and all(_is_neurovec(x) for x in img):
            return _make_neurovec_adapter(img, mask=mask, tr=tr, start_time=start_time)
        if _is_nifti1image(first) and all(_is_nifti1image(x) for x in img):
            return _make_nibabel_adapter(img, mask=mask, tr=tr, start_time=start_time)
        if isinstance(first, (str, Path)) and all(
            isinstance(x, (str, Path)) for x in img
        ):
            return _make_paths_adapter(img, mask=mask, tr=tr, start_time=start_time)
        raise TypeError(
            "fmri_dataset: list/tuple input must be homogeneous "
            "(NeuroVec, Nifti1Image, or path)"
        )

    # Single-value dispatch
    if _is_neurovec(img):
        return _make_neurovec_adapter(img, mask=mask, tr=tr, start_time=start_time)
    if _is_nifti1image(img):
        return _make_nibabel_adapter(img, mask=mask, tr=tr, start_time=start_time)
    if isinstance(img, (str, Path)):
        return _make_paths_adapter(img, mask=mask, tr=tr, start_time=start_time)
    if isinstance(img, np.ndarray):
        if img.ndim == 4:
            return _make_array4d_adapter(img, mask=mask, tr=tr, start_time=start_time)
        if img.ndim == 2:
            return _make_matrix_adapter(
                img,
                mask=mask,
                tr=tr,
                start_time=start_time,
                run_length=run_length,
            )

    raise TypeError(
        f"fmri_dataset: cannot build adapter from {type(img).__name__!r}; "
        "supported types: NeuroVec, Nifti1Image, str/Path, 4-D or 2-D ndarray, "
        "or any object satisfying DatasetProtocol"
    )


def _require_tr(
    tr: float | Sequence[float] | None, *, what: str
) -> float | Sequence[float]:
    if tr is None:
        raise ValueError(f"`tr` is required for {what} input")
    if isinstance(tr, (int, float)):
        return float(tr)
    return tr


def _make_neurovec_adapter(
    img: object,
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
) -> DatasetProtocol:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter(
        img, mask=mask, tr=_require_tr(tr, what="NeuroVec"), start_time=start_time
    )


def _make_nibabel_adapter(
    img: object,
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
) -> DatasetProtocol:
    from .adapters.nibabel_adapter import NibabelAdapter

    images = img if isinstance(img, (list, tuple)) else [img]
    return NibabelAdapter(
        images,
        mask=mask,
        tr=_require_tr(tr, what="Nifti1Image"),
        start_time=start_time,
    )


def _make_paths_adapter(
    paths: object,
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
) -> DatasetProtocol:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter.from_paths(
        cast(Any, paths),
        mask=mask,
        tr=_require_tr(tr, what="path"),
        start_time=start_time,
    )


def _make_array4d_adapter(
    arr: NDArray[np.float64],
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
) -> DatasetProtocol:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter.from_array(
        arr,
        mask=mask,
        tr=_require_tr(tr, what="4-D ndarray"),
        start_time=start_time,
    )


def _make_matrix_adapter(
    arr: NDArray[np.float64],
    *,
    mask: object,
    tr: float | Sequence[float] | None,
    start_time: float | Sequence[float] | None,
    run_length: int | Sequence[int] | None = None,
) -> DatasetProtocol:
    resolved_tr = _require_tr(tr, what="2-D matrix")
    n_rows = int(arr.shape[0])
    if run_length is None:
        blocklens = [n_rows]
    else:
        blocklens = _normalize_run_lengths(run_length, n_rows=n_rows)
    sampling_frame = SamplingFrame(
        blocklens=blocklens, tr=resolved_tr, start_time=start_time
    )
    flat_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    backend = matrix_backend(np.asarray(arr, dtype=np.float64), mask=flat_mask)
    return BackendAdapter(backend, sampling_frame)


def _resolve_tr(
    tr: float | list[float] | None,
    TR: float | list[float] | None,
) -> float | list[float]:
    """Resolve canonical and compatibility TR spellings."""
    if tr is None and TR is None:
        raise ValueError("matrix_dataset requires tr or TR")
    if tr is not None and TR is not None:
        if not same_seconds(tr, TR):
            raise ValueError("tr and TR must agree when both are supplied")
    resolved = TR if tr is None else tr
    assert resolved is not None
    return resolved


def _resolve_start_time(
    start_time: float | Sequence[float] | None,
    slice_timing_offset: float | Sequence[float] | None,
) -> float | Sequence[float] | None:
    """Resolve canonical / discoverable spellings of the sampling-grid offset.

    ``start_time`` and ``slice_timing_offset`` are interchangeable. If
    both are passed they must agree; either being ``None`` defers to the
    other. Returning ``None`` lets :class:`SamplingFrame` apply its
    own default (``TR/2`` — the BOLD-midpoint convention).
    """
    if start_time is None and slice_timing_offset is None:
        return None
    if start_time is not None and slice_timing_offset is not None:
        if not same_seconds(start_time, slice_timing_offset):
            raise ValueError(
                "start_time and slice_timing_offset must agree when both "
                "are supplied"
            )
    resolved = slice_timing_offset if start_time is None else start_time
    # A dataset's first-volume offset is a time since the run began
    # (fmridataset 99d19fe); SamplingFrame itself stays permissive.
    if np.any(np.asarray(resolved, dtype=np.float64) < 0):
        raise ValueError("start_time must be non-negative")
    return resolved


def _normalize_run_lengths(
    run_length: int | Sequence[int],
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
