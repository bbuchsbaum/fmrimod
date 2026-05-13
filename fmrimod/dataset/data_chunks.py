"""Canonical typed chunk iteration over fMRI datasets."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .fmri_dataset import FmriDataset


@dataclass(frozen=True)
class DataChunk:
    """A single chunk of data extracted from a dataset."""

    data: NDArray[np.floating[Any]]
    voxel_ind: NDArray[np.intp]
    row_ind: NDArray[np.intp]
    chunk_num: int

    def __repr__(self) -> str:
        return (
            f"DataChunk(chunk_num={self.chunk_num}, "
            f"shape={self.data.shape}, "
            f"n_voxels={len(self.voxel_ind)}, "
            f"n_rows={len(self.row_ind)})"
        )


def data_chunk(
    mat: NDArray[np.floating[Any]],
    voxel_ind: NDArray[np.intp],
    row_ind: NDArray[np.intp],
    chunk_num: int,
) -> DataChunk:
    """Create a :class:`DataChunk`."""
    return DataChunk(
        data=np.asarray(mat),
        voxel_ind=np.asarray(voxel_ind, dtype=np.intp),
        row_ind=np.asarray(row_ind, dtype=np.intp),
        chunk_num=int(chunk_num),
    )


class ChunkIterator:
    """Iterator that yields :class:`DataChunk` objects."""

    def __init__(self, nchunks: int, get_chunk: Callable[[int], DataChunk]) -> None:
        self._nchunks = int(nchunks)
        self._get_chunk = get_chunk
        self._current = 0

    @property
    def nchunks(self) -> int:
        """Number of chunks this iterator produces."""
        return self._nchunks

    def __len__(self) -> int:
        return self._nchunks

    def __repr__(self) -> str:
        return f"ChunkIterator(nchunks={self._nchunks}, current={self._current})"

    def __iter__(self) -> Iterator[DataChunk]:
        self._current = 0
        return self

    def __next__(self) -> DataChunk:
        if self._current >= self._nchunks:
            raise StopIteration
        chunk = self._get_chunk(self._current + 1)
        self._current += 1
        return chunk

    def nextElem(self) -> DataChunk:  # noqa: N802
        """R-iterator compatibility alias for :func:`next`."""
        return next(self)

    def collect(self) -> list[DataChunk]:
        """Materialize all chunks."""
        return list(self)


def data_chunks(
    dataset: FmriDataset,
    nchunks: int = 1,
    *,
    runwise: bool = False,
) -> ChunkIterator:
    """Create a typed chunk iterator for a dataset."""
    if not isinstance(dataset, FmriDataset):
        raise TypeError("data_chunks expects a fmrimod.dataset.FmriDataset")
    if runwise:
        return _runwise_chunks(dataset)

    nchunks = _validate_nchunks(nchunks)
    n_voxels = int(dataset.n_voxels)
    if nchunks > n_voxels:
        warnings.warn(
            f"requested {nchunks} chunks but only {n_voxels} voxels; "
            f"using {n_voxels} chunks instead",
            stacklevel=2,
        )
        nchunks = n_voxels

    splits = _split_indices(n_voxels, nchunks)
    row_ind = np.arange(dataset.n_timepoints, dtype=np.intp)
    voxel_ind = _dataset_voxel_indices(dataset)

    def get_chunk(chunk_num: int) -> DataChunk:
        col_idx = splits[chunk_num - 1]
        return DataChunk(
            data=dataset.get_data(rows=row_ind, cols=col_idx),
            voxel_ind=voxel_ind[col_idx],
            row_ind=row_ind,
            chunk_num=chunk_num,
        )

    return ChunkIterator(nchunks, get_chunk)


def voxel_index_chunks(
    x: object,
    nchunks: int | None = None,
    chunk_size: int | None = None,
) -> list[NDArray[np.intp]]:
    """Return bare voxel-index chunks for legacy callers."""
    if hasattr(x, "n_voxels"):
        n_voxels = int(x.n_voxels)
    else:
        arr = np.asarray(x)
        if arr.ndim < 1:
            raise ValueError("x must be an array or dataset-like object")
        n_voxels = int(arr.shape[-1])

    if chunk_size is None:
        if nchunks is None:
            nchunks = 1
        chunk_size = int(np.ceil(n_voxels / max(1, int(nchunks))))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return [
        np.arange(start, min(start + chunk_size, n_voxels), dtype=np.intp)
        for start in range(0, n_voxels, chunk_size)
    ]


def collect_chunks(iterator: ChunkIterator) -> list[DataChunk]:
    """Materialize all chunks from *iterator* into a list."""
    return iterator.collect()


def exec_strategy(
    strategy: Literal["voxelwise", "runwise", "chunkwise"] = "voxelwise",
    nchunks: int | None = None,
) -> Callable[[FmriDataset], ChunkIterator]:
    """Create a reusable chunking strategy."""
    if strategy not in {"voxelwise", "runwise", "chunkwise"}:
        raise ValueError("strategy must be 'voxelwise', 'runwise', or 'chunkwise'")
    if strategy == "chunkwise" and nchunks is None:
        raise ValueError("nchunks is required for chunkwise strategy")
    if strategy == "chunkwise":
        assert nchunks is not None
        nchunks = _validate_nchunks(nchunks)

    def apply(dataset: FmriDataset) -> ChunkIterator:
        if strategy == "runwise":
            return data_chunks(dataset, runwise=True)
        n_voxels = int(dataset.n_voxels)
        if strategy == "voxelwise":
            return data_chunks(dataset, nchunks=n_voxels)
        assert nchunks is not None
        return data_chunks(dataset, nchunks=min(nchunks, n_voxels))

    return apply


def _runwise_chunks(dataset: FmriDataset) -> ChunkIterator:
    voxel_ind = _dataset_voxel_indices(dataset)

    def get_chunk(chunk_num: int) -> DataChunk:
        run = chunk_num - 1
        row_ind = _run_row_indices(dataset, run)
        return DataChunk(
            data=dataset.get_run_data(run),
            voxel_ind=voxel_ind,
            row_ind=row_ind,
            chunk_num=chunk_num,
        )

    return ChunkIterator(dataset.n_runs, get_chunk)


def _dataset_voxel_indices(dataset: FmriDataset) -> NDArray[np.intp]:
    mask = dataset.get_mask()
    if mask.dtype == np.bool_:
        return np.where(mask)[0].astype(np.intp)
    return np.arange(dataset.n_voxels, dtype=np.intp)


def _run_row_indices(dataset: FmriDataset, run: int) -> NDArray[np.intp]:
    start = int(np.sum(dataset.run_lengths[:run]))
    stop = start + int(dataset.run_lengths[run])
    return np.arange(start, stop, dtype=np.intp)


def _validate_nchunks(nchunks: int) -> int:
    if isinstance(nchunks, bool) or not isinstance(nchunks, Integral):
        raise ValueError("nchunks must be a positive integer")
    value = int(nchunks)
    if value <= 0:
        raise ValueError("nchunks must be a positive integer")
    return value


def _split_indices(n: int, nchunks: int) -> list[NDArray[np.intp]]:
    assignments = np.sort(
        np.tile(np.arange(nchunks), (n + nchunks - 1) // nchunks)[:n]
    )
    return [np.where(assignments == i)[0].astype(np.intp) for i in range(nchunks)]
