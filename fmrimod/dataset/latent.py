"""Canonical latent-component dataset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..sampling import SamplingFrame
from .backend_protocol import StorageBackend
from .backends.latent_backend import InMemoryLatentBackend, LatentBackend
from .errors import ConfigError


@dataclass
class LatentDataset:
    """Dataset backed by a latent score/loadings storage backend."""

    backend: StorageBackend
    sampling_frame: SamplingFrame
    event_table: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        dims = self.backend.get_dims()
        if sum(self.sampling_frame.blocklens) != dims.time:
            raise ValueError("sampling frame length must match latent score rows")

    @property
    def scores(self) -> NDArray[np.float64]:
        """Latent scores in ``timepoints x components`` orientation."""
        return np.asarray(self.backend.get_data(), dtype=np.float64)

    @property
    def loadings(self) -> NDArray[np.float64]:
        """Spatial loadings in ``voxels x components`` orientation."""
        return self.get_spatial_loadings()

    @property
    def n_runs(self) -> int:
        return len(self.sampling_frame.blocklens)

    @property
    def n_timepoints(self) -> list[int]:
        return [int(v) for v in self.sampling_frame.blocklens]

    @property
    def n_voxels(self) -> int:
        return int(self.backend.get_dims().spatial[0])

    @property
    def storage_backend(self) -> StorageBackend:
        return self.backend

    def _run_rows(self, run: int) -> NDArray[np.intp]:
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"run must be within [0, {self.n_runs - 1}]")
        start = int(sum(self.sampling_frame.blocklens[:run]))
        end = start + int(self.sampling_frame.blocklens[run])
        return np.arange(start, end, dtype=np.intp)

    def get_latent_scores(
        self,
        *,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return latent scores with optional row/component selection."""
        return np.asarray(self.backend.get_data(rows=rows, cols=cols), dtype=np.float64)

    def get_scores(self, run: int = 0) -> NDArray[np.float64]:
        """Return latent scores for one run."""
        return self.get_latent_scores(rows=self._run_rows(run))

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
        *,
        rows: NDArray[np.intp] | None = None,
        voxels: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return reconstructed voxel-space data."""
        method = getattr(self.backend, "reconstruct_voxels", None)
        if method is None:
            raise ConfigError("latent backend does not expose reconstruct_voxels")
        return np.asarray(method(rows=rows, voxels=voxels), dtype=np.float64)

    def get_data(self, run: int = 0) -> NDArray[np.float64]:
        """Return reconstructed voxel data for one run."""
        return self.reconstruct_voxels(rows=self._run_rows(run))

    def get_data_matrix(
        self,
        *,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return reconstructed voxel data across all runs."""
        return self.reconstruct_voxels(rows=rows, voxels=cols)

    def get_mask(self) -> NDArray[np.bool_]:
        return self.backend.get_mask().reshape((self.n_voxels, 1, 1))

    def get_sampling_frame(self) -> SamplingFrame:
        return self.sampling_frame

    def get_censor(self, run: int = 0) -> None:
        return None

    def get_component_info(self) -> dict[str, Any]:
        """Return latent component metadata."""
        return self.backend.get_metadata()


def _run_lengths_from_scores(
    n_rows: int,
    run_length: int | Sequence[int] | None,
) -> list[int]:
    if run_length is None:
        return [int(n_rows)]
    if isinstance(run_length, int):
        if n_rows % int(run_length):
            raise ValueError("run_length must divide score rows")
        return [int(run_length)] * (n_rows // int(run_length))
    blocklens = [int(v) for v in run_length]
    if sum(blocklens) != n_rows:
        raise ValueError("sum(run_length) must equal score rows")
    return blocklens


def _source_paths(
    source: str | Path | Sequence[str | Path],
    base_path: str | Path,
) -> str | Path | list[Path]:
    if isinstance(source, (str, Path)):
        src = Path(source)
        return src if src.is_absolute() else Path(base_path) / src
    paths: list[Path] = []
    for item in source:
        src = Path(item)
        paths.append(src if src.is_absolute() else Path(base_path) / src)
    return paths


def latent_dataset(
    scores: ArrayLike | None = None,
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
) -> LatentDataset:
    """Construct the canonical latent dataset.

    Array inputs use the same latent backend contract in memory. ``source=``
    creates the storage-backed HDF5 latent backend ported from fmridataset-py.
    """
    resolved_tr = float(tr if TR is None else TR)
    if source is not None:
        backend = LatentBackend(_source_paths(source, base_path), preload=preload)
        backend.open()
        run_lengths = (
            backend.run_lengths
            if run_length is None
            else _run_lengths_from_scores(backend.get_dims().time, run_length)
        )
    else:
        if scores is None:
            raise TypeError("latent_dataset() requires scores or source=")
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim != 2:
            raise ValueError("scores must be a 2-D matrix")
        run_lengths = _run_lengths_from_scores(scores_arr.shape[0], run_length)
        loadings_arr = (
            None if loadings is None else np.asarray(loadings, dtype=np.float64)
        )
        if loadings_arr is not None and loadings_arr.ndim == 2:
            if loadings_arr.shape[0] == scores_arr.shape[1]:
                loadings_arr = loadings_arr.T
        backend = InMemoryLatentBackend(
            scores_arr,
            loadings=loadings_arr,
            offset=None if offset is None else np.asarray(offset, dtype=np.float64),
            run_lengths=run_lengths,
        )
        backend.open()

    sf = SamplingFrame(blocklens=run_lengths, tr=resolved_tr)
    return LatentDataset(backend=backend, sampling_frame=sf, event_table=event_table)


def get_latent_scores(
    dataset: LatentDataset,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Return latent scores from a :class:`LatentDataset`."""
    return dataset.get_latent_scores(rows=rows, cols=cols)


def get_spatial_loadings(
    dataset: LatentDataset,
    components: NDArray[np.intp] | Sequence[int] | int | None = None,
) -> NDArray[np.float64]:
    """Return spatial loadings from a :class:`LatentDataset`."""
    return dataset.get_spatial_loadings(components=components)


def get_component_info(dataset: LatentDataset) -> dict[str, Any]:
    """Return latent component metadata."""
    return dataset.get_component_info()


__all__ = [
    "LatentDataset",
    "get_component_info",
    "get_latent_scores",
    "get_spatial_loadings",
    "latent_dataset",
]
