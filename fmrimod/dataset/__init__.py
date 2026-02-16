"""Dataset abstractions for fMRI data.

Provides protocol definitions and concrete implementations for accessing
fMRI time-series data in a format-agnostic way.
"""

from .protocols import DatasetProtocol, MaskProtocol, ChunkIterator
from .fmri_dataset import FmriDataset
from .chunking import VoxelChunker, BlockChunker

__all__ = [
    "DatasetProtocol",
    "MaskProtocol",
    "ChunkIterator",
    "FmriDataset",
    "VoxelChunker",
    "BlockChunker",
]
