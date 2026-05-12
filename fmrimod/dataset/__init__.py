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
from .compat import (
    LatentDataset,
    create_design_matrix_from_benchmark,
    data_chunks,
    design_plot,
    evaluate_method_performance,
    extract_csv_data,
    fmri_latent_lm,
    fmri_mem_dataset,
    get_benchmark_summary,
    latent_dataset,
    list_benchmark_datasets,
    load_benchmark_dataset,
    read_fmri_config,
    read_h5_full,
    read_nifti_full,
    register_basis,
    resolve_basis,
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
    "LatentDataset",
    "fmri_mem_dataset",
    "latent_dataset",
    "fmri_latent_lm",
    "data_chunks",
    "extract_csv_data",
    "read_h5_full",
    "read_nifti_full",
    "read_fmri_config",
    "register_basis",
    "resolve_basis",
    "load_benchmark_dataset",
    "list_benchmark_datasets",
    "get_benchmark_summary",
    "create_design_matrix_from_benchmark",
    "evaluate_method_performance",
    "design_plot",
]
