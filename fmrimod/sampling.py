"""Sampling frame for fMRI acquisition timing specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(init=False)
class SamplingFrame:
    """fMRI acquisition timing specification.

    This class represents the temporal sampling structure of an fMRI experiment,
    including the timing of acquisitions, block structure, and precision settings.

    Attributes:
        blocklens: Number of scans per block (array or scalar)
        tr: Repetition time(s) in seconds (array or scalar)
        start_time: Start time offset(s) in seconds (array or scalar, default TR/2)
        precision: Temporal precision for convolution in seconds
    """
    blocklens: Union[ArrayLike, int]
    tr: Union[ArrayLike, float]
    start_time: Union[ArrayLike, float] = None
    precision: float = 0.1

    def __init__(
        self,
        blocklens=None,
        tr=None,
        start_time=None,
        precision=0.1,
        *,
        n_scans=None,
        TR=None,
    ):
        # Support legacy pyfmridesign names: n_scans -> blocklens, TR -> tr
        if blocklens is None and n_scans is not None:
            blocklens = n_scans
        if tr is None and TR is not None:
            tr = TR
        if tr is not None and TR is not None:
            tr_cmp = np.atleast_1d(np.asarray(tr, dtype=float))
            tr_alias_cmp = np.atleast_1d(np.asarray(TR, dtype=float))
            same_shape = tr_cmp.shape == tr_alias_cmp.shape
            same_values = same_shape and np.allclose(tr_cmp, tr_alias_cmp)
            if not same_values:
                raise TypeError("cannot provide both `tr` and `TR` with different values")
        if blocklens is None or tr is None:
            raise TypeError("SamplingFrame requires blocklens (or n_scans) and tr (or TR)")
        self.blocklens = blocklens
        self.tr = tr
        self.start_time = start_time
        self.precision = precision
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate and normalize inputs."""
        # Convert to numpy arrays
        self.blocklens = np.atleast_1d(self.blocklens).astype(int)
        self.tr = np.atleast_1d(self.tr).astype(float)
        
        # Validate inputs
        if np.any(self.blocklens <= 0):
            raise ValueError("blocklens must be positive")
        if np.any(self.tr <= 0):
            raise ValueError("tr must be positive")
        if self.precision <= 0:
            raise ValueError("precision must be positive")
        
        # Expand scalars to match number of blocks
        n_blocks = len(self.blocklens)
        
        if len(self.tr) == 1 and n_blocks > 1:
            self.tr = np.repeat(self.tr, n_blocks)
        elif len(self.tr) != n_blocks:
            raise ValueError(
                f"Length of tr ({len(self.tr)}) must be 1 or match "
                f"number of blocks ({n_blocks})"
            )

        # Precision must be smaller than every TR.
        min_tr = float(np.min(self.tr))
        if self.precision >= min_tr:
            raise ValueError("precision must be less than the minimum TR")

        # Default start_time is TR/2 for each block.
        if self.start_time is None:
            self.start_time = self.tr / 2.0
        else:
            self.start_time = np.atleast_1d(self.start_time)
            if len(self.start_time) == 1 and n_blocks > 1:
                self.start_time = np.repeat(self.start_time, n_blocks)
            elif len(self.start_time) != n_blocks:
                raise ValueError(
                    f"Length of start_time ({len(self.start_time)}) must be 1 or match "
                    f"number of blocks ({n_blocks})"
                )
    
    @property
    def TR(self) -> float:
        """Repetition time (scalar, for backward compatibility)."""
        return float(self.tr[0])

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        return len(self.blocklens)
    
    @property
    def n_scans(self) -> int:
        """Total number of scans across all blocks."""
        return int(np.sum(self.blocklens))
    
    @property
    def blockids(self) -> NDArray[np.int32]:
        """Block ID for each scan.
        
        Returns:
            Array of 0-based block IDs with length equal to total scans
        """
        ids = []
        for i, block_len in enumerate(self.blocklens):
            ids.extend([i] * block_len)
        return np.array(ids, dtype=np.int32)
    
    @property
    def samples(self) -> NDArray[np.float64]:
        """Time points for each scan.
        
        Returns:
            Array of time points in seconds for each scan
        """
        return self.sample_times(global_time=True)
    
    def sample_times(self, global_time: bool = True, blockids: Optional[ArrayLike] = None) -> NDArray[np.float64]:
        """Get time points for scans with optional filtering.
        
        Args:
            global_time: If True, return global times; if False, return block-relative times
            blockids: Optional 0-based block IDs to filter.
            
        Returns:
            Array of time points in seconds
        """
        if blockids is not None:
            blockids = np.atleast_1d(blockids)
            if np.any(blockids < 0) or np.any(blockids >= self.n_blocks):
                raise ValueError(f"blockids must be in range [0, {self.n_blocks - 1}]")
            block_indices = blockids.astype(np.int64)
        else:
            block_indices = np.arange(self.n_blocks, dtype=np.int64)

        if len(block_indices) == 0:
            return np.array([], dtype=np.float64)

        times = []
        block_durations = self.blocklens * self.tr
        cumulative_time = np.concatenate([[0.0], np.cumsum(block_durations)])
        
        for i in block_indices:
            block_len = int(self.blocklens[i])
            tr = float(self.tr[i])
            start = float(self.start_time[i])
            block_times = np.arange(block_len) * tr + start

            if global_time:
                block_times = cumulative_time[i] + block_times

            times.extend(block_times)
        
        return np.array(times, dtype=np.float64)
    
    @property
    def acquisition_onsets(self) -> NDArray[np.float64]:
        """Get fMRI acquisition onset times.
        
        Calculate the onset time in seconds for each fMRI volume acquisition
        from the start of the experiment.
        
        This returns the temporal onset of each brain volume acquisition, accounting
        for TR, start_time, and run structure. This is essentially a convenience
        wrapper around sample_times(global_time=True) that provides clearer
        semantic meaning for the common use case of getting acquisition times.
        
        Note: The onset times include the start_time offset (default TR/2),
        so the first acquisition typically doesn't start at 0.
        
        Returns:
            Array of acquisition onset times in seconds
        """
        return self.sample_times(global_time=True)
    
    def global_onsets(self, onsets: ArrayLike, blockids: ArrayLike) -> NDArray[np.float64]:
        """Convert block-relative onset times to global (experiment-wide) onset times.

        Args:
            onsets: Numeric array of onset times within blocks (block-relative).
            blockids: Integer array identifying which block each onset belongs to
                (0-based).

        Returns:
            Array of global onset times.

        Raises:
            ValueError: If onsets and blockids have different lengths or blockids are out of range.
        """
        onsets = np.asarray(onsets, dtype=np.float64)
        blockids = np.asarray(blockids, dtype=np.int64)

        if len(onsets) != len(blockids):
            raise ValueError("onsets and blockids must have the same length")

        if len(blockids) > 0:
            if np.any(blockids < 0) or np.any(blockids >= len(self.blocklens)):
                raise ValueError(f"blockids must be in range [0, {len(self.blocklens) - 1}]")

        # Calculate cumulative time offsets for each block
        block_durations = self.blocklens * self.tr
        cumulative_time = np.concatenate([[0.0], np.cumsum(block_durations)])

        return onsets + cumulative_time[blockids]

    @property
    def global_scan_times(self) -> NDArray[np.float64]:
        """Global scan times across all blocks.

        Returns:
            Array of absolute time points for each scan
        """
        return self.samples
    
    def block_samples(self, block_idx: int) -> NDArray[np.float64]:
        """Get time points for a specific block.
        
        Args:
            block_idx: Block index (0-based)
            
        Returns:
            Array of time points for the specified block
        """
        if block_idx < 0 or block_idx >= self.n_blocks:
            raise ValueError(f"block_idx must be in range [0, {self.n_blocks-1}]")
        
        block_len = self.blocklens[block_idx]
        tr = self.tr[block_idx]
        start = self.start_time[block_idx]
        
        return start + np.arange(block_len) * tr
    
    def __str__(self) -> str:
        """String representation of SamplingFrame."""
        lines = [
            f"SamplingFrame with {self.n_blocks} block(s) and {self.n_scans} total scans:",
        ]
        
        for i in range(self.n_blocks):
            lines.append(
                f"  Block {i}: {self.blocklens[i]} scans, "
                f"tr={self.tr[i]}s, start={self.start_time[i]}s"
            )
        
        lines.append(f"  Precision: {self.precision}s")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation of SamplingFrame."""
        return (
            f"SamplingFrame(blocklens={self.blocklens.tolist()}, "
            f"tr={self.tr.tolist()}, start_time={self.start_time.tolist()}, "
            f"precision={self.precision})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all sampling frame parameters
        """
        return {
            'blocklens': self.blocklens.tolist(),
            'tr': self.tr.tolist(),
            'start_time': self.start_time.tolist(),
            'precision': self.precision,
            'n_blocks': self.n_blocks,
            'n_scans': self.n_scans,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SamplingFrame:
        """Create SamplingFrame from dictionary.
        
        Args:
            data: Dictionary with sampling frame parameters
            
        Returns:
            New SamplingFrame instance
        """
        return cls(
            blocklens=data['blocklens'],
            tr=data.get('tr', data.get('TR')),
            start_time=data.get('start_time', 0.0),
            precision=data.get('precision', 0.1),
        )
    
    def concatenate(self, other: SamplingFrame) -> SamplingFrame:
        """Concatenate two sampling frames.
        
        Args:
            other: Another SamplingFrame to concatenate
            
        Returns:
            New SamplingFrame with combined blocks
        """
        if self.precision != other.precision:
            raise ValueError(
                f"Cannot concatenate SamplingFrames with different precision "
                f"({self.precision} vs {other.precision})"
            )
        
        # Adjust start times for the second frame
        last_end_time = self.samples[-1] + self.tr[self.n_blocks - 1]
        adjusted_start_times = other.start_time + last_end_time
        
        return SamplingFrame(
            blocklens=np.concatenate([self.blocklens, other.blocklens]),
            tr=np.concatenate([self.tr, other.tr]),
            start_time=np.concatenate([self.start_time, adjusted_start_times]),
            precision=self.precision,
        )
