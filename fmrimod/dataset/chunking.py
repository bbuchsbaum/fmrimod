"""Chunk iterators for memory-efficient voxel processing."""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray


class VoxelChunker:
    """Iterate over contiguous chunks of voxels.

    Parameters
    ----------
    n_voxels : int
        Total number of voxels.
    chunk_size : int
        Number of voxels per chunk.  The last chunk may be smaller.
    """

    def __init__(self, n_voxels: int, chunk_size: int = 5000):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self._n_voxels = n_voxels
        self._chunk_size = chunk_size

    def iter_chunks(
        self, data: NDArray[np.float64]
    ) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.intp]]]:
        """Yield ``(data_chunk, voxel_indices)`` from a ``(time, voxels)`` matrix."""
        for start in range(0, self._n_voxels, self._chunk_size):
            end = min(start + self._chunk_size, self._n_voxels)
            indices = np.arange(start, end, dtype=np.intp)
            yield data[:, start:end], indices

    @property
    def n_chunks(self) -> int:
        return int(np.ceil(self._n_voxels / self._chunk_size))


class BlockChunker:
    """Iterate over predefined blocks (e.g., ROIs) of voxels.

    Parameters
    ----------
    labels : NDArray[np.intp]
        1-D array of length ``n_voxels`` assigning each voxel to a block.
        Unique label values define the blocks.
    """

    def __init__(self, labels: NDArray[np.intp]):
        self._labels = np.asarray(labels, dtype=np.intp)
        self._unique_labels = np.unique(self._labels)

    def iter_chunks(
        self, data: NDArray[np.float64]
    ) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.intp]]]:
        """Yield ``(data_chunk, voxel_indices)`` for each block label."""
        for label in self._unique_labels:
            indices = np.where(self._labels == label)[0].astype(np.intp)
            yield data[:, indices], indices

    @property
    def n_chunks(self) -> int:
        return len(self._unique_labels)
