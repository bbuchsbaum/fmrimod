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

from fmrimod.sampling import SamplingFrame

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
    events: pd.DataFrame | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
    start_time: float | Sequence[float] | None = None,
    slice_timing_offset: float | Sequence[float] | None = None,
    run_length: int | Sequence[int] | None = None,
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
        Optional event table.
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
    data_source
        Legacy alias accepting a :class:`DatasetProtocol`-compatible object.
    event_table
        Legacy alias for ``events``.
    """
    resolved_events = _resolve_events(events, event_table)
    resolved_start_time = _resolve_start_time(start_time, slice_timing_offset)

    if data_source is not None:
        if img is not None:
            raise ValueError("Pass `img` or `data_source`, not both")
        if run_length is not None:
            raise ValueError("`run_length` is not valid with `data_source`")
        return FmriDataset(data_source, event_table=resolved_events, censor=censor)

    if img is None:
        raise ValueError("fmri_dataset requires `img` (or legacy `data_source`)")

    # Adapter chain expects a scalar start_time; ``None`` means "use the
    # SamplingFrame default" which the adapter currently encodes as 0.0
    # (frame-start). To keep the typed dataset constructor's documented
    # default of TR/2 (BOLD-midpoint), we resolve None against the
    # supplied TR right here.
    adapter_start: float
    if resolved_start_time is None:
        tr_scalar = float(np.atleast_1d(np.asarray(tr, dtype=float))[0])
        adapter_start = tr_scalar / 2.0
    else:
        adapter_start = float(np.atleast_1d(np.asarray(resolved_start_time))[0])
    adapter = _build_adapter(
        img, mask=mask, tr=tr, start_time=adapter_start, run_length=run_length
    )
    return FmriDataset(adapter, event_table=resolved_events, censor=censor)


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
    events: pd.DataFrame | None,
    event_table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if events is not None and event_table is not None and event_table is not events:
        raise ValueError("Pass `events` or `event_table`, not both")
    return events if events is not None else event_table


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
    start_time: float,
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


def _require_tr(tr: float | Sequence[float] | None, *, what: str) -> float:
    if tr is None:
        raise ValueError(f"`tr` is required for {what} input")
    if isinstance(tr, (int, float)):
        return float(tr)
    return tr  # type: ignore[return-value]


def _make_neurovec_adapter(
    img: object, *, mask: object, tr: float | Sequence[float] | None, start_time: float
) -> DatasetProtocol:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter(
        img, mask=mask, tr=_require_tr(tr, what="NeuroVec"), start_time=start_time
    )


def _make_nibabel_adapter(
    img: object, *, mask: object, tr: float | Sequence[float] | None, start_time: float
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
    paths: object, *, mask: object, tr: float | Sequence[float] | None, start_time: float
) -> DatasetProtocol:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter.from_paths(
        cast(Any, paths),
        mask=mask,
        tr=_require_tr(tr, what="path"),
        start_time=start_time,
    )


def _make_array4d_adapter(
    arr: NDArray[np.float64], *, mask: object, tr: float | Sequence[float] | None, start_time: float
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
    start_time: float,
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
        if not np.array_equal(np.asarray(tr), np.asarray(TR)):
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
        if not np.array_equal(
            np.asarray(start_time, dtype=float),
            np.asarray(slice_timing_offset, dtype=float),
        ):
            raise ValueError(
                "start_time and slice_timing_offset must agree when both "
                "are supplied"
            )
    return slice_timing_offset if start_time is None else start_time


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
