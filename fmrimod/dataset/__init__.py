"""Dataset abstractions for fMRI data.

Provides protocol definitions and concrete implementations for accessing
fMRI time-series data in a format-agnostic way.
"""

from .protocols import DatasetProtocol, MaskProtocol, ChunkIterator
from .fmri_dataset import FmriDataset
from .chunking import VoxelChunker, BlockChunker
from .group_data import (
    GroupData,
    group_data,
    group_data_from_csv,
    group_data_from_fmrilm,
    group_data_from_h5,
    group_data_from_nifti,
    detect_group_data_format,
)

__all__ = [
    "DatasetProtocol",
    "MaskProtocol",
    "ChunkIterator",
    "FmriDataset",
    "VoxelChunker",
    "BlockChunker",
    "GroupData",
    "group_data",
    "group_data_from_csv",
    "group_data_from_fmrilm",
    "group_data_from_h5",
    "group_data_from_nifti",
    "detect_group_data_format",
]
