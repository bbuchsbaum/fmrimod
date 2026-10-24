"""Concrete FmriDataset implementation.

A dataset is one storage backend (``time x in-mask voxels``, read by
``rows``/``cols``), one :class:`SamplingFrame` that splits its rows into
runs, an optional event table and optional censoring. Runs are row ranges
of the backend, so every read goes through one contract and fetches only
what it asks for.
"""

from __future__ import annotations

from typing import List, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..sampling import SamplingFrame
from .backend_protocol import BackendDims, StorageBackend
from .protocols import DatasetProtocol
from .space import VolumeSpace

DataSource = Union[StorageBackend, DatasetProtocol]


def _resolve_source(
    data_source: object, sampling_frame: SamplingFrame | None
) -> tuple[StorageBackend, SamplingFrame]:
    """Return the canonical backend and sampling frame behind ``data_source``."""
    if isinstance(data_source, StorageBackend):
        if sampling_frame is None:
            raise TypeError("a StorageBackend source needs a sampling_frame")
        return data_source, sampling_frame
    frame = sampling_frame
    if frame is None:
        get_frame = getattr(data_source, "get_sampling_frame", None)
        if not callable(get_frame):
            raise TypeError(
                "FmriDataset expects a StorageBackend (with sampling_frame=) or an "
                f"object satisfying DatasetProtocol; got {type(data_source).__name__}"
            )
        frame = cast(SamplingFrame, get_frame())
    # fmrimod's own adapters are thin facades over a backend.
    backend = getattr(data_source, "backend", None)
    if isinstance(backend, StorageBackend):
        return backend, frame
    from .backends.runwise_backend import RunwiseSourceBackend

    return RunwiseSourceBackend(data_source), frame


def check_run_lengths(backend: StorageBackend, frame: SamplingFrame) -> None:
    """Refuse a sampling frame that does not fit the backend's rows.

    The total must match, and when the backend reports one length per
    source and the frame has as many runs, each run must match its source
    (fmridataset 631cf0d, #81): a frame of ``[7, 6]`` over files of ``[6, 7]``
    rows would pull file 2's first volume into run 1.
    """
    n_rows = int(backend.get_dims().time)
    if n_rows != int(frame.n_scans):
        raise ValueError(
            f"backend has {n_rows} timepoints but the sampling frame has "
            f"{int(frame.n_scans)}"
        )
    lengths = backend.source_run_lengths
    blocklens = [int(v) for v in frame.blocklens]
    if lengths is not None and len(lengths) == len(blocklens):
        for i, (want, have) in enumerate(zip(blocklens, lengths, strict=True)):
            if want != have:
                raise ValueError(
                    f"Run {i} has {want} volumes in the sampling frame but its "
                    f"source holds {have}"
                )


class FmriDataset:
    """fMRI dataset combining time-series data with experimental events.

    Parameters
    ----------
    data_source : StorageBackend or DatasetProtocol
        A storage backend (pass ``sampling_frame``), one of fmrimod's
        adapters, or any object satisfying :class:`DatasetProtocol`.
    event_table : pd.DataFrame, optional
        Table of experimental events with at least ``onset`` and ``block``
        (or ``run``) columns.  Additional columns are event variables.
    censor : NDArray[np.bool_] or list of NDArray, optional
        Boolean censoring vector(s).  ``True`` marks volumes to exclude.
        Can be a single array covering all runs concatenated, or a list
        of per-run arrays.
    sampling_frame : SamplingFrame, optional
        Run structure of the rows. Required with a bare backend; otherwise
        it overrides the adapter's own frame.
    confounds : list of pd.DataFrame, optional
        Per-run nuisance tables (e.g. fMRIPrep ``*_desc-confounds_timeseries``),
        one row per scan. Nothing is modelled from them by default; a
        ``confounds("trans_x", ...)`` spec term without ``source=`` selects
        columns from these tables when the model is fitted.
    """

    def __init__(
        self,
        data_source: DataSource,
        event_table: Optional[pd.DataFrame] = None,
        censor: Optional[Union[NDArray[np.bool_], List[NDArray[np.bool_]]]] = None,
        *,
        sampling_frame: Optional[SamplingFrame] = None,
        confounds: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        backend, frame = _resolve_source(data_source, sampling_frame)
        check_run_lengths(backend, frame)
        self._source = data_source
        self._backend = backend
        self._frame = frame
        starts = np.concatenate([[0], np.cumsum(frame.blocklens)]).astype(np.intp)
        self._run_starts = starts
        self._event_table = event_table
        self._censor = self._validate_censor(censor)
        self._confounds = self._validate_confounds(confounds)

    # -- Data access --

    def get_data(
        self,
        run: Optional[int] = None,
        *,
        rows: Optional[NDArray[np.intp]] = None,
        cols: Optional[NDArray[np.intp]] = None,
    ) -> NDArray[np.float64]:
        """Return data by matrix slice or, temporarily, by run.

        ``rows``/``cols`` select timepoints and in-mask voxels; the backend
        reads only those. Positional run access is retained during the
        migration; new code should use :meth:`get_run_data`.
        """
        if rows is not None or cols is not None:
            return np.asarray(self._backend.get_data(rows=rows, cols=cols), dtype=np.float64)
        if run is None:
            return self.get_all_data()
        return self.get_run_data(run)

    def get_run_data(self, run: int) -> NDArray[np.float64]:
        """Return ``(time, voxels)`` data for one run."""
        return np.asarray(self._backend.get_data(rows=self.run_rows(run)), dtype=np.float64)

    def run_rows(self, run: int) -> NDArray[np.intp]:
        """Row indices of ``run`` (0-based) in the concatenated data."""
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"Run {run} out of range [0, {self.n_runs})")
        return np.arange(self._run_starts[run], self._run_starts[run + 1], dtype=np.intp)

    def get_all_data(self) -> NDArray[np.float64]:
        """Return all runs vertically concatenated as ``(total_time, voxels)``."""
        return np.asarray(self._backend.get_data(), dtype=np.float64)

    def get_data_matrix(
        self,
        *,
        rows: Optional[NDArray[np.intp]] = None,
        cols: Optional[NDArray[np.intp]] = None,
    ) -> NDArray[np.float64]:
        """Return all data as a ``timepoints x voxels`` matrix."""
        if rows is None and cols is None:
            return self.get_all_data()
        return self.get_data(rows=rows, cols=cols)

    # -- Space --

    @property
    def space(self) -> VolumeSpace:
        """Spatial identity of the columns: grid, in-mask voxels, affine."""
        return self._backend.space

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the mask: in the grid's shape for images, flat for matrices."""
        return self.space.mask_array()

    def get_affine(self) -> NDArray[np.float64] | None:
        """Return the 4x4 voxel-to-world affine, or ``None`` when unknown."""
        affine = self.space.affine
        return None if affine is None else affine.copy()

    def get_dims(self) -> BackendDims:
        """Return backend dimensions."""
        return self._backend.get_dims()

    # -- Timing --

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the :class:`SamplingFrame`."""
        return self._frame

    @property
    def sampling_frame(self) -> SamplingFrame:
        """Convenience alias for ``get_sampling_frame()``."""
        return self._frame

    @property
    def storage_backend(self) -> StorageBackend:
        """The storage backend every read goes through."""
        return self._backend

    @property
    def n_runs(self) -> int:
        return int(self._frame.n_blocks)

    @property
    def n_timepoints(self) -> int:
        """Total number of timepoints across all runs."""
        return int(sum(self.run_lengths))

    @property
    def run_lengths(self) -> List[int]:
        """Number of timepoints per run."""
        return [int(v) for v in self._frame.blocklens]

    @property
    def blocklens(self) -> List[int]:
        """Alias for per-run lengths during the consolidation migration."""
        return self.run_lengths

    @property
    def n_voxels(self) -> int:
        return int(self.space.n_voxels)

    # -- Event table --

    @property
    def event_table(self) -> Optional[pd.DataFrame]:
        """The event table, or *None* if not set."""
        return self._event_table

    def with_event_table(self, event_table: pd.DataFrame) -> "FmriDataset":
        """Return this dataset with a replacement event table."""
        return FmriDataset(
            self._backend,
            event_table=event_table,
            censor=self._censor,
            sampling_frame=self._frame,
            confounds=self._confounds,
        )

    # -- Confounds --

    @property
    def confounds(self) -> Optional[List[pd.DataFrame]]:
        """Per-run nuisance tables attached to the dataset, or *None*."""
        return self._confounds

    def _validate_confounds(
        self, confounds: Optional[Union[pd.DataFrame, List[pd.DataFrame]]]
    ) -> Optional[List[pd.DataFrame]]:
        if confounds is None:
            return None
        lengths = [int(n) for n in self.run_lengths]
        if isinstance(confounds, pd.DataFrame):
            if len(confounds) != sum(lengths):
                raise ValueError(
                    f"confounds has {len(confounds)} rows; the dataset has "
                    f"{sum(lengths)} scans"
                )
            bounds = np.cumsum([0, *lengths])
            return [
                confounds.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True)
                for i in range(len(lengths))
            ]
        frames = list(confounds)
        if len(frames) != len(lengths):
            raise ValueError(
                f"{len(frames)} confounds tables for {len(lengths)} runs"
            )
        for run, (frame, n) in enumerate(zip(frames, lengths, strict=True)):
            if len(frame) != n:
                raise ValueError(
                    f"confounds table for run {run + 1} has {len(frame)} rows; "
                    f"the run has {n} scans"
                )
        return [frame.reset_index(drop=True) for frame in frames]

    # -- Censoring --

    @property
    def censor(self) -> Optional[List[NDArray[np.bool_]]]:
        """Per-run boolean censor vectors, or *None*."""
        return self._censor

    def get_censor(self, run: int) -> Optional[NDArray[np.bool_]]:
        """Return the censor vector for a single run, or *None*."""
        if self._censor is None:
            return None
        return self._censor[run]

    def __repr__(self) -> str:
        return (
            f"FmriDataset(n_runs={self.n_runs}, "
            f"n_timepoints={self.n_timepoints}, "
            f"n_voxels={self.n_voxels})"
        )

    # -- Internal --

    def _validate_censor(
        self, censor: Optional[Union[NDArray[np.bool_], List[NDArray[np.bool_]]]]
    ) -> Optional[List[NDArray[np.bool_]]]:
        if censor is None:
            return None

        n_tp = self.run_lengths

        if isinstance(censor, np.ndarray) and censor.ndim == 1:
            # Split a single concatenated vector into per-run vectors
            total = sum(n_tp)
            if len(censor) != total:
                raise ValueError(
                    f"Censor vector length ({len(censor)}) "
                    f"!= total timepoints ({total})"
                )
            splits = np.cumsum(n_tp[:-1])
            return [c.astype(bool) for c in np.split(censor, splits)]

        # Already a list
        censor_list = list(censor)
        if len(censor_list) != self.n_runs:
            raise ValueError(
                f"Number of censor arrays ({len(censor_list)}) "
                f"!= n_runs ({self.n_runs})"
            )
        for i, (c, nt) in enumerate(zip(censor_list, n_tp)):
            c = np.asarray(c, dtype=bool)
            if len(c) != nt:
                raise ValueError(
                    f"Run {i}: censor length ({len(c)}) != n_timepoints ({nt})"
                )
            censor_list[i] = c
        return censor_list
