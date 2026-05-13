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
from typing import Any, Sequence

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
    img: Any = None,
    *,
    mask: Any = None,
    tr: float | Sequence[float] | None = None,
    events: pd.DataFrame | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
    start_time: float = 0.0,
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
    start_time
        Onset of the first volume (defaults to 0.0).
    data_source
        Legacy alias accepting a :class:`DatasetProtocol`-compatible object.
    event_table
        Legacy alias for ``events``.
    """
    resolved_events = _resolve_events(events, event_table)

    if data_source is not None:
        if img is not None:
            raise ValueError("Pass `img` or `data_source`, not both")
        return FmriDataset(data_source, event_table=resolved_events, censor=censor)

    if img is None:
        raise ValueError("fmri_dataset requires `img` (or legacy `data_source`)")

    adapter = _build_adapter(img, mask=mask, tr=tr, start_time=start_time)
    return FmriDataset(adapter, event_table=resolved_events, censor=censor)


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
    return FmriDataset(adapter, event_table=event_table)


# -- Internal helpers ----------------------------------------------------------


def _resolve_events(
    events: pd.DataFrame | None,
    event_table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if events is not None and event_table is not None and event_table is not events:
        raise ValueError("Pass `events` or `event_table`, not both")
    return events if events is not None else event_table


def _is_dataset_protocol(obj: Any) -> bool:
    """Duck-type check; avoid runtime_checkable Protocol cost for hot path."""
    return (
        hasattr(obj, "get_data")
        and hasattr(obj, "get_mask")
        and hasattr(obj, "get_sampling_frame")
        and hasattr(obj, "n_runs")
    )


def _is_nifti1image(obj: Any) -> bool:
    try:
        import nibabel  # type: ignore[import-not-found]
    except ImportError:
        return False
    return isinstance(obj, nibabel.Nifti1Image)


def _build_adapter(
    img: Any,
    *,
    mask: Any,
    tr: float | Sequence[float] | None,
    start_time: float,
) -> Any:
    # DatasetProtocol-compatible: use as-is.
    if _is_dataset_protocol(img):
        return img

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
            return _make_matrix_adapter(img, mask=mask, tr=tr)

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
    img: Any, *, mask: Any, tr: Any, start_time: float
) -> Any:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter(
        img, mask=mask, tr=_require_tr(tr, what="NeuroVec"), start_time=start_time
    )


def _make_nibabel_adapter(
    img: Any, *, mask: Any, tr: Any, start_time: float
) -> Any:
    from .adapters.nibabel_adapter import NibabelAdapter

    images = img if isinstance(img, (list, tuple)) else [img]
    return NibabelAdapter(
        images,
        mask=mask,
        tr=_require_tr(tr, what="Nifti1Image"),
        start_time=start_time,
    )


def _make_paths_adapter(
    paths: Any, *, mask: Any, tr: Any, start_time: float
) -> Any:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter.from_paths(
        paths,
        mask=mask,
        tr=_require_tr(tr, what="path"),
        start_time=start_time,
    )


def _make_array4d_adapter(
    arr: NDArray, *, mask: Any, tr: Any, start_time: float
) -> Any:
    from .adapters.neuroim_adapter import NeuroVecAdapter

    return NeuroVecAdapter.from_array(
        arr,
        mask=mask,
        tr=_require_tr(tr, what="4-D ndarray"),
        start_time=start_time,
    )


def _make_matrix_adapter(arr: NDArray, *, mask: Any, tr: Any) -> Any:
    resolved_tr = _require_tr(tr, what="2-D matrix")
    sampling_frame = SamplingFrame(blocklens=[int(arr.shape[0])], tr=resolved_tr)
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
