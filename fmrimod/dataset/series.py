"""Canonical fMRI time-series container and query helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .fmri_dataset import FmriDataset
from .selectors import SeriesSelector, resolve_indices


class FmriSeries:
    """Container for selected fMRI time-series data."""

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        voxel_info: pd.DataFrame,
        temporal_info: pd.DataFrame,
        selection_info: dict[str, object] | None = None,
        dataset_info: dict[str, object] | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self._voxel_info = voxel_info
        self._temporal_info = temporal_info
        self._selection_info = selection_info or {}
        self._dataset_info = dataset_info or {}

        if self._data.ndim != 2:
            raise ValueError("data must be a 2-D array")
        if len(voxel_info) != self._data.shape[1]:
            raise ValueError(
                f"voxel_info rows ({len(voxel_info)}) must equal "
                f"data columns ({self._data.shape[1]})"
            )
        if len(temporal_info) != self._data.shape[0]:
            raise ValueError(
                f"temporal_info rows ({len(temporal_info)}) must equal "
                f"data rows ({self._data.shape[0]})"
            )

    @property
    def data(self) -> NDArray[np.floating[Any]]:
        return self._data

    @property
    def voxel_info(self) -> pd.DataFrame:
        return self._voxel_info

    @property
    def temporal_info(self) -> pd.DataFrame:
        return self._temporal_info

    @property
    def selection_info(self) -> dict[str, object]:
        return self._selection_info

    @property
    def dataset_info(self) -> dict[str, object]:
        return self._dataset_info

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._data.shape[0]), int(self._data.shape[1]))

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        """Return the series data as a dense array."""
        return np.array(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a long-form DataFrame with one row per voxel and timepoint."""
        n_time, n_vox = self._data.shape
        time_idx = np.repeat(np.arange(n_time), n_vox)
        vox_idx = np.tile(np.arange(n_vox), n_time)
        out = pd.concat(
            [
                self._temporal_info.iloc[time_idx].reset_index(drop=True),
                self._voxel_info.iloc[vox_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        out["signal"] = self._data.ravel()
        return out

    def __repr__(self) -> str:
        backend = self._dataset_info.get("backend_type", "?")
        return (
            f"<FmriSeries {self.shape[1]} voxels x {self.shape[0]} timepoints | "
            f"backend={backend}>"
        )

    def __len__(self) -> int:
        return self.shape[0]


def new_fmri_series(
    data: NDArray[np.floating[Any]],
    voxel_info: pd.DataFrame,
    temporal_info: pd.DataFrame,
    selection_info: dict[str, object] | None = None,
    dataset_info: dict[str, object] | None = None,
) -> FmriSeries:
    """Construct an :class:`FmriSeries` from explicit components."""
    return FmriSeries(data, voxel_info, temporal_info, selection_info, dataset_info)


def is_fmri_series(obj: object) -> bool:
    """Return whether *obj* is an :class:`FmriSeries`."""
    return isinstance(obj, FmriSeries)


def as_matrix(obj: FmriSeries) -> NDArray[np.floating[Any]]:
    """Return an :class:`FmriSeries` as a dense NumPy matrix."""
    if not isinstance(obj, FmriSeries):
        raise TypeError("obj must be an FmriSeries")
    return obj.to_numpy()


def to_dataframe(obj: FmriSeries) -> pd.DataFrame:
    """Return an :class:`FmriSeries` as a long-form pandas DataFrame."""
    if not isinstance(obj, FmriSeries):
        raise TypeError("obj must be an FmriSeries")
    return obj.to_dataframe()


def as_tibble(obj: FmriSeries) -> pd.DataFrame:
    """Compatibility alias for :func:`to_dataframe`."""
    return to_dataframe(obj)


def resolve_selector(
    dataset: FmriDataset,
    selector: SeriesSelector | NDArray[np.intp] | NDArray[np.bool_] | None,
) -> NDArray[np.intp]:
    """Resolve a spatial selector to masked-space column indices."""
    return resolve_indices(selector, dataset)


def resolve_timepoints(
    dataset: FmriDataset,
    timepoints: NDArray[np.intp] | NDArray[np.bool_] | None,
) -> NDArray[np.intp]:
    """Resolve a temporal selector to 0-based row indices."""
    n_time = dataset.n_timepoints
    if timepoints is None:
        return np.arange(n_time, dtype=np.intp)

    arr = np.asarray(timepoints)
    if np.issubdtype(arr.dtype, np.bool_):
        if arr.size != n_time:
            raise ValueError(
                f"Boolean timepoints length ({arr.size}) must equal "
                f"n_timepoints ({n_time})"
            )
        return np.flatnonzero(arr).astype(np.intp)
    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(np.intp, copy=False).ravel()
        if np.any(idx < 0) or np.any(idx >= n_time):
            raise IndexError(f"timepoints out of range [0, {n_time})")
        return idx
    raise ValueError(f"Unsupported timepoints type: {arr.dtype}")


def fmri_series(
    dataset: FmriDataset,
    selector: SeriesSelector | NDArray[np.intp] | NDArray[np.bool_] | None = None,
    timepoints: NDArray[np.intp] | NDArray[np.bool_] | None = None,
    output: str = "fmri_series",
    event_window: object | None = None,
) -> FmriSeries | NDArray[np.floating[Any]]:
    """Query fMRI time-series from a canonical dataset."""
    del event_window
    voxel_ind = resolve_selector(dataset, selector)
    time_ind = resolve_timepoints(dataset, timepoints)
    data = dataset.get_data(rows=time_ind, cols=voxel_ind)

    if output in {"matrix", "ndarray"}:
        return data
    if output != "fmri_series":
        raise ValueError("output must be one of {'fmri_series', 'matrix', 'ndarray'}")

    return FmriSeries(
        data=data,
        voxel_info=_build_voxel_info(dataset, voxel_ind),
        temporal_info=_build_temporal_info(dataset, time_ind),
        selection_info={
            "selector": repr(selector) if selector is not None else None,
            "timepoints": repr(timepoints) if timepoints is not None else None,
        },
        dataset_info={"backend_type": _backend_type(dataset)},
    )


def series(
    dataset: FmriDataset,
    selector: SeriesSelector | NDArray[np.intp] | NDArray[np.bool_] | None = None,
    timepoints: NDArray[np.intp] | NDArray[np.bool_] | None = None,
    output: str = "fmri_series",
    event_window: object | None = None,
) -> FmriSeries | NDArray[np.floating[Any]]:
    """Compatibility alias for :func:`fmri_series`."""
    warnings.warn(
        "series() is deprecated; use fmri_series() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fmri_series(dataset, selector, timepoints, output, event_window)


def _build_voxel_info(
    dataset: FmriDataset,
    voxel_indices: NDArray[np.intp],
) -> pd.DataFrame:
    mask = dataset.get_mask().ravel()
    full_indices = np.flatnonzero(mask)[voxel_indices]
    dims = dataset.get_dims().spatial
    x = full_indices % dims[0] + 1
    y = (full_indices // dims[0]) % dims[1] + 1
    z = full_indices // (dims[0] * dims[1]) + 1
    return pd.DataFrame(
        {
            "voxel": voxel_indices,
            "linear_index": full_indices.astype(np.intp),
            "x": x.astype(np.intp),
            "y": y.astype(np.intp),
            "z": z.astype(np.intp),
        }
    )


def _build_temporal_info(
    dataset: FmriDataset,
    time_indices: NDArray[np.intp],
) -> pd.DataFrame:
    frame = dataset.sampling_frame
    return pd.DataFrame(
        {
            "run_id": frame.run_ids(one_based=False)[time_indices],
            "timepoint": time_indices,
            "sample_time": frame.samples[time_indices],
        }
    )


def _backend_type(dataset: FmriDataset) -> str:
    backend = dataset.storage_backend
    if backend is None:
        return type(dataset).__name__
    return type(backend).__name__
