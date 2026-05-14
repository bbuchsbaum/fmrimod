"""Canonical latent-component dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fmrimod.sampling import SamplingFrame

from .backend_protocol import StorageBackend
from .backends.latent_backend import InMemoryLatentBackend, LatentBackend
from .errors import ConfigError
from .fmri_dataset import FmriDataset


class _LatentAdapter:
    """Adapt latent backends to the current run-wise dataset protocol."""

    def __init__(self, backend: StorageBackend, sampling_frame: SamplingFrame) -> None:
        self.backend = backend
        self._sampling_frame = sampling_frame

    def _run_rows(self, run: int) -> NDArray[np.intp]:
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"Run {run} out of range [0, {self.n_runs})")
        start = int(np.sum(self.run_lengths[:run]))
        end = start + int(self.run_lengths[run])
        return np.arange(start, end, dtype=np.intp)

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return reconstructed voxel data for one run."""
        method = getattr(self.backend, "reconstruct_voxels", None)
        if method is None:
            raise ConfigError("latent backend does not expose reconstruct_voxels")
        data = method(rows=self._run_rows(run))
        return np.asarray(data, dtype=np.float64)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the voxel mask from the latent backend."""
        return np.asarray(self.backend.get_mask(), dtype=np.bool_)

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame."""
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return int(self._sampling_frame.n_blocks)

    @property
    def run_lengths(self) -> list[int]:
        return [int(v) for v in self._sampling_frame.blocklens]

    @property
    def n_timepoints(self) -> list[int]:
        return self.run_lengths

    @property
    def n_voxels(self) -> int:
        return int(self.backend.get_mask().sum())


class LatentDataset(FmriDataset):
    """Dataset backed by a latent score/loadings storage backend.

    Positional/run-wise data access returns reconstructed voxel-space data,
    matching the historic ``latent_dataset(scores, loadings=...)`` behavior.
    Latent-space operations are explicit through :meth:`get_latent_scores`,
    :meth:`get_spatial_loadings`, and :meth:`reconstruct_voxels`.
    """

    def __init__(
        self,
        backend: StorageBackend,
        sampling_frame: SamplingFrame,
        event_table: pd.DataFrame | None = None,
        censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
    ) -> None:
        if backend.get_dims().time != sampling_frame.n_scans:
            raise ValueError("sampling frame length must match latent score rows")
        super().__init__(
            _LatentAdapter(backend, sampling_frame),
            event_table=event_table,
            censor=censor,
        )

    @property
    def backend(self) -> StorageBackend:
        """Underlying latent storage backend."""
        backend = self.storage_backend
        if backend is None:  # pragma: no cover - impossible for _LatentAdapter
            raise ConfigError("latent dataset has no storage backend")
        return backend

    @property
    def TR(self) -> float:  # noqa: N802
        """First repetition time in seconds."""
        return self.sampling_frame.TR

    @property
    def scores(self) -> NDArray[np.float64]:
        """Latent scores in ``timepoints x components`` orientation."""
        return self.get_latent_scores()

    @property
    def loadings(self) -> NDArray[np.float64]:
        """Spatial loadings in ``voxels x components`` orientation."""
        return self.get_spatial_loadings()

    def get_latent_scores(
        self,
        rows: NDArray[np.intp] | Sequence[int] | int | None = None,
        cols: NDArray[np.intp] | Sequence[int] | int | None = None,
    ) -> NDArray[np.float64]:
        """Return latent scores with optional row/component selection."""
        return np.asarray(self.backend.get_data(rows=rows, cols=cols), dtype=np.float64)

    def get_scores(self, run: int = 0) -> NDArray[np.float64]:
        """Return latent scores for one run."""
        rows = self._source._run_rows(run)
        return self.get_latent_scores(rows=rows)

    def get_spatial_loadings(
        self,
        components: NDArray[np.intp] | Sequence[int] | int | None = None,
    ) -> NDArray[np.float64]:
        """Return spatial loadings with optional component selection."""
        method = getattr(self.backend, "get_loadings", None)
        if method is None:
            raise ConfigError("latent backend does not expose get_loadings")
        return np.asarray(method(components=components), dtype=np.float64)

    def reconstruct_voxels(
        self,
        rows: NDArray[np.intp] | Sequence[int] | int | None = None,
        voxels: NDArray[np.intp] | Sequence[int] | int | None = None,
    ) -> NDArray[np.float64]:
        """Return reconstructed voxel-space data."""
        method = getattr(self.backend, "reconstruct_voxels", None)
        if method is None:
            raise ConfigError("latent backend does not expose reconstruct_voxels")
        return np.asarray(method(rows=rows, voxels=voxels), dtype=np.float64)

    def get_component_info(self) -> dict[str, object]:
        """Return latent component metadata."""
        return self.backend.get_metadata()

    def with_event_table(self, event_table: pd.DataFrame) -> LatentDataset:
        """Return this latent dataset with a replacement event table."""
        return LatentDataset(
            self.backend,
            self.sampling_frame,
            event_table=event_table,
            censor=self.censor,
        )

    def __repr__(self) -> str:
        meta = self.get_component_info()
        return (
            "LatentDataset("
            f"n_runs={self.n_runs}, "
            f"n_timepoints={self.n_timepoints}, "
            f"n_voxels={self.n_voxels}, "
            f"n_components={meta.get('n_components', '?')}"
            ")"
        )


def latent_dataset(
    scores: ArrayLike | str | Path | Sequence[str | Path] | None = None,
    loadings: ArrayLike | None = None,
    tr: float = 1.0,
    run_length: int | Sequence[int] | None = None,
    *,
    source: str | Path | Sequence[str | Path] | None = None,
    TR: float | None = None,  # noqa: N803
    base_path: str | Path = ".",
    event_table: pd.DataFrame | None = None,
    preload: bool = False,
    offset: ArrayLike | None = None,
    censor: NDArray[np.bool_] | list[NDArray[np.bool_]] | None = None,
) -> LatentDataset:
    """Construct the canonical latent dataset.

    ``scores``/``loadings`` inputs use the same backend contract in memory.
    ``source=`` creates a storage-backed HDF5 latent backend with ``basis`` and
    ``loadings`` datasets.
    """
    resolved_tr = _resolve_tr(tr, TR)
    event_table = _validate_event_table(event_table)

    if source is None and _looks_like_source(scores) and loadings is None:
        source = scores  # type: ignore[assignment]
        scores = None

    if source is not None:
        if scores is not None:
            raise TypeError("pass either scores or source=, not both")
        backend = LatentBackend(_source_paths(source, base_path), preload=preload)
        backend.open()
        run_lengths = _normalize_run_lengths(
            backend.get_dims().time,
            run_length,
            inferred=backend.run_lengths,
            noun="number of time points",
        )
    else:
        if scores is None:
            raise TypeError("latent_dataset() requires scores or source=")
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim != 2:
            raise ValueError("scores must be a 2-D matrix")
        run_lengths = _normalize_run_lengths(
            int(scores_arr.shape[0]),
            run_length,
            noun="score rows",
        )
        backend = InMemoryLatentBackend(
            scores_arr,
            loadings=_coerce_loadings(scores_arr, loadings),
            offset=None if offset is None else np.asarray(offset, dtype=np.float64),
            run_lengths=run_lengths,
        )
        backend.open()

    sf = SamplingFrame(blocklens=run_lengths, tr=resolved_tr)
    return LatentDataset(
        backend,
        sampling_frame=sf,
        event_table=event_table,
        censor=censor,
    )


def get_latent_scores(
    dataset: LatentDataset,
    rows: NDArray[np.intp] | Sequence[int] | int | None = None,
    cols: NDArray[np.intp] | Sequence[int] | int | None = None,
) -> NDArray[np.float64]:
    """Return latent scores from a :class:`LatentDataset`."""
    return dataset.get_latent_scores(rows=rows, cols=cols)


def get_spatial_loadings(
    dataset: LatentDataset,
    components: NDArray[np.intp] | Sequence[int] | int | None = None,
) -> NDArray[np.float64]:
    """Return spatial loadings from a :class:`LatentDataset`."""
    return dataset.get_spatial_loadings(components=components)


def get_component_info(dataset: LatentDataset) -> dict[str, object]:
    """Return latent component metadata."""
    return dataset.get_component_info()


def _resolve_tr(tr: float, TR: float | None) -> float:  # noqa: N803
    if TR is None:
        return float(tr)
    if tr != 1.0 and not np.isclose(float(tr), float(TR)):
        raise ValueError("tr and TR must agree when both are supplied")
    return float(TR)


def _validate_event_table(event_table: pd.DataFrame | None) -> pd.DataFrame | None:
    if event_table is not None and not event_table.columns.is_unique:
        raise ValueError("event_table columns must be unique")
    return event_table


def _looks_like_source(obj: object) -> bool:
    if isinstance(obj, (str, Path)):
        return True
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
        return bool(obj) and all(isinstance(item, (str, Path)) for item in obj)
    return False


def _source_paths(
    source: str | Path | Sequence[str | Path],
    base_path: str | Path,
) -> str | Path | list[Path]:
    base = Path(base_path)
    if isinstance(source, (str, Path)):
        src = Path(source)
        return src if src.is_absolute() else base / src
    paths: list[Path] = []
    for item in source:
        if not isinstance(item, (str, Path)):
            raise ConfigError("source entries must be strings or Path objects")
        src = Path(item)
        paths.append(src if src.is_absolute() else base / src)
    if not paths:
        raise ConfigError("source must contain at least one path", parameter="source")
    return paths


def _normalize_run_lengths(
    n_rows: int,
    run_length: int | Sequence[int] | None,
    *,
    inferred: Sequence[int] | None = None,
    noun: str,
) -> list[int]:
    if run_length is None:
        return [int(v) for v in inferred] if inferred is not None else [int(n_rows)]
    if isinstance(run_length, (int, np.integer)):
        value = int(run_length)
        if value == 0 and inferred is not None:
            return [int(v) for v in inferred]
        if value <= 0:
            raise ValueError("run_length values must be positive")
        if n_rows % value:
            raise ValueError("run_length must divide score rows")
        return [value] * (n_rows // value)

    values = list(run_length)
    if not values or sum(float(v) for v in values) == 0:
        if inferred is None:
            raise ValueError("run_length values must be positive")
        return [int(v) for v in inferred]
    for value in values:
        if not float(value).is_integer():
            raise ValueError("run_length values must be integers")
    blocklens = [int(v) for v in values]
    if any(v <= 0 for v in blocklens):
        raise ValueError("run_length values must be positive")
    if sum(blocklens) != n_rows:
        raise ValueError(f"sum(run_length) must equal {noun}")
    return blocklens


def _coerce_loadings(
    scores: NDArray[np.float64],
    loadings: ArrayLike | None,
) -> NDArray[np.float64] | None:
    if loadings is None:
        return None
    arr = np.asarray(loadings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("loadings must be a 2-D matrix")
    n_components = int(scores.shape[1])
    if arr.shape[0] == n_components:
        return arr.T
    if arr.shape[1] == n_components:
        return arr
    raise ValueError("loadings rows or columns must match score columns")


__all__ = [
    "LatentDataset",
    "get_component_info",
    "get_latent_scores",
    "get_spatial_loadings",
    "latent_dataset",
]
