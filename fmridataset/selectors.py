"""Compatibility re-exports for dataset selectors."""

from fmrimod.dataset.selectors import (
    AllSelector,
    IndexSelector,
    MaskSelector,
    ROISelector,
    SeriesSelector,
    SphereSelector,
    VoxelSelector,
    all_selector,
    index_selector,
    mask_selector,
    resolve_indices,
    roi_selector,
    sphere_selector,
    voxel_selector,
)

__all__ = [
    "SeriesSelector",
    "IndexSelector",
    "AllSelector",
    "ROISelector",
    "VoxelSelector",
    "SphereSelector",
    "MaskSelector",
    "index_selector",
    "all_selector",
    "roi_selector",
    "voxel_selector",
    "sphere_selector",
    "mask_selector",
    "resolve_indices",
]
