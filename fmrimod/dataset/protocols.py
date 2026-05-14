"""Protocol definitions for fMRI dataset abstractions.

Defines the interfaces that dataset implementations must satisfy,
enabling format-agnostic access to fMRI time-series data.
"""

from __future__ import annotations

from typing import Iterator, List, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..sampling import SamplingFrame


@runtime_checkable
class MaskProtocol(Protocol):
    """Protocol for spatial masks.

    A mask selects a subset of voxels from a 3-D volume.
    """

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the 3-D boolean mask array."""
        ...

    @property
    def n_voxels(self) -> int:
        """Number of ``True`` voxels in the mask."""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the 3-D volume."""
        ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for fMRI dataset objects.

    Any object satisfying this protocol can be passed to :func:`fmri_lm`
    for GLM fitting.  Implementations must provide per-run data access,
    mask information, and temporal sampling metadata.
    """

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return the data matrix for a single run.

        Parameters
        ----------
        run : int
            Zero-indexed run number.

        Returns
        -------
        NDArray
            Array of shape ``(n_timepoints, n_voxels)``.
        """
        ...

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the 3-D boolean spatial mask."""
        ...

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the :class:`SamplingFrame` for this dataset."""
        ...

    @property
    def n_runs(self) -> int:
        """Number of runs in the dataset."""
        ...

    @property
    def n_timepoints(self) -> List[int]:
        """Number of time points per run."""
        ...

    @property
    def n_voxels(self) -> int:
        """Number of voxels (columns) in each data matrix."""
        ...


class ChunkIterator(Protocol):
    """Protocol for iterating over voxel chunks.

    Chunk iterators enable memory-efficient processing of large datasets
    by yielding subsets of voxels at a time.
    """

    def __iter__(self) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.intp]]]:
        """Yield ``(data_chunk, voxel_indices)`` pairs.

        Each ``data_chunk`` has shape ``(n_timepoints, chunk_size)`` and
        ``voxel_indices`` is a 1-D integer array of the corresponding
        column indices into the full data matrix.
        """
        ...

    @property
    def n_chunks(self) -> int:
        """Total number of chunks that will be yielded."""
        ...
