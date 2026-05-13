"""Spatial selectors for canonical fMRI datasets."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fmri_dataset import FmriDataset


class SeriesSelector(ABC):
    """Base class for spatial selectors."""

    @abstractmethod
    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        """Resolve to 0-based column indices within masked data."""


class IndexSelector(SeriesSelector):
    """Select voxels by 0-based column indices within masked data."""

    def __init__(self, indices: NDArray[np.intp] | list[int]) -> None:
        self.indices = np.asarray(indices, dtype=np.intp).ravel()

    def __repr__(self) -> str:
        preview = self.indices[:5].tolist()
        suffix = "..." if self.indices.size > 5 else ""
        return f"IndexSelector(indices={preview}{suffix})"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        n_voxels = int(dataset.n_voxels)
        if np.any(self.indices < 0) or np.any(self.indices >= n_voxels):
            raise IndexError(f"Index out of range [0, {n_voxels})")
        return self.indices.astype(np.intp, copy=False)


class AllSelector(SeriesSelector):
    """Select all voxels within the dataset mask."""

    def __repr__(self) -> str:
        return "AllSelector()"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        return np.arange(dataset.n_voxels, dtype=np.intp)


class ROISelector(SeriesSelector):
    """Select voxels falling within a full-volume ROI mask."""

    def __init__(self, roi_mask: NDArray[np.bool_]) -> None:
        self.roi_mask = np.asarray(roi_mask, dtype=np.bool_).ravel()

    def __repr__(self) -> str:
        return f"ROISelector(n_selected={int(self.roi_mask.sum())}, size={self.roi_mask.size})"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dataset_mask = dataset.get_mask().ravel()
        if self.roi_mask.size != dataset_mask.size:
            raise ValueError(
                f"ROI mask length ({self.roi_mask.size}) must equal "
                f"dataset mask length ({dataset_mask.size})"
            )

        dataset_voxels = np.flatnonzero(dataset_mask)
        selected = np.isin(dataset_voxels, np.flatnonzero(self.roi_mask))
        return np.flatnonzero(selected).astype(np.intp)


class VoxelSelector(SeriesSelector):
    """Select voxels by 1-based ``(x, y, z)`` coordinates."""

    def __init__(self, coords: NDArray[np.intp] | list[list[int]]) -> None:
        arr = np.asarray(coords, dtype=np.intp)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError("Single coordinate must have length 3")
            arr = arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.coords = arr

    def __repr__(self) -> str:
        preview = self.coords[:3].tolist()
        suffix = "..." if self.coords.shape[0] > 3 else ""
        return f"VoxelSelector(coords={preview}{suffix})"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dims = dataset.get_dims().spatial
        mask = dataset.get_mask().ravel()
        for axis, dim_size in enumerate(dims):
            if np.any(self.coords[:, axis] < 1) or np.any(
                self.coords[:, axis] > dim_size
            ):
                raise IndexError(f"Coordinate axis {axis} out of range [1, {dim_size}]")

        linear = (
            (self.coords[:, 0] - 1)
            + (self.coords[:, 1] - 1) * dims[0]
            + (self.coords[:, 2] - 1) * dims[0] * dims[1]
        )
        return _full_indices_to_masked_columns(mask, linear)


class SphereSelector(SeriesSelector):
    """Select voxels within a sphere in 1-based voxel coordinates."""

    def __init__(
        self,
        center: NDArray[np.floating[Any]] | list[float] | tuple[float, ...],
        radius: float,
    ) -> None:
        c = np.asarray(center, dtype=np.float64)
        if c.size != 3:
            raise ValueError("center must have length 3")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.center = c
        self.radius = float(radius)

    def __repr__(self) -> str:
        return f"SphereSelector(center={self.center.tolist()}, radius={self.radius})"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dims = dataset.get_dims().spatial
        mask = dataset.get_mask().ravel()
        zz, yy, xx = np.meshgrid(
            np.arange(1, dims[2] + 1),
            np.arange(1, dims[1] + 1),
            np.arange(1, dims[0] + 1),
            indexing="ij",
        )
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        dist = np.sqrt(np.sum((coords - self.center) ** 2, axis=1))
        full_indices = np.flatnonzero(dist <= self.radius).astype(np.intp)
        selected = _full_indices_to_masked_columns(mask, full_indices, allow_empty=True)
        if selected.size == 0:
            raise ValueError("Spherical ROI does not overlap with dataset mask")
        return selected


class MaskSelector(SeriesSelector):
    """Select voxels by a boolean mask in full-volume or masked space."""

    def __init__(self, mask: NDArray[np.bool_]) -> None:
        self.mask = np.asarray(mask, dtype=np.bool_).ravel()

    def __repr__(self) -> str:
        return f"MaskSelector(n_selected={int(self.mask.sum())}, size={self.mask.size})"

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dataset_mask = dataset.get_mask().ravel()
        n_volume = dataset_mask.size
        n_masked = int(dataset_mask.sum())

        if self.mask.size == n_masked:
            cols = np.flatnonzero(self.mask).astype(np.intp)
        elif self.mask.size == n_volume:
            dataset_voxels = np.flatnonzero(dataset_mask)
            selected = np.isin(dataset_voxels, np.flatnonzero(self.mask))
            cols = np.flatnonzero(selected).astype(np.intp)
        else:
            raise ValueError(
                f"Mask length ({self.mask.size}) does not match "
                f"volume size ({n_volume}) or masked size ({n_masked})"
            )

        if cols.size == 0:
            raise ValueError("Mask selector selected no voxels")
        return cols


def index_selector(indices: NDArray[np.intp] | list[int]) -> IndexSelector:
    """Create an index selector for masked-space voxel columns."""
    return IndexSelector(indices)


def all_selector() -> AllSelector:
    """Create a selector for all voxels."""
    return AllSelector()


def roi_selector(roi: NDArray[np.bool_]) -> ROISelector:
    """Create a selector from a full-volume ROI mask."""
    return ROISelector(roi)


def voxel_selector(coords: NDArray[np.intp] | list[list[int]]) -> VoxelSelector:
    """Create a selector from 1-based voxel coordinates."""
    return VoxelSelector(coords)


def sphere_selector(
    center: NDArray[np.floating[Any]] | list[float] | tuple[float, ...],
    radius: float,
) -> SphereSelector:
    """Create a spherical ROI selector."""
    return SphereSelector(center=center, radius=radius)


def mask_selector(mask: NDArray[np.bool_]) -> MaskSelector:
    """Create a selector from a logical mask."""
    return MaskSelector(mask)


def resolve_indices(
    selector: SeriesSelector | NDArray[np.intp] | NDArray[np.bool_] | None,
    dataset: FmriDataset,
) -> NDArray[np.intp]:
    """Resolve a selector against *dataset*."""
    if selector is None:
        return np.arange(dataset.n_voxels, dtype=np.intp)
    if isinstance(selector, SeriesSelector):
        return selector.resolve_indices(dataset)

    arr = np.asarray(selector)
    if np.issubdtype(arr.dtype, np.bool_):
        return MaskSelector(arr).resolve_indices(dataset)
    if np.issubdtype(arr.dtype, np.integer):
        return IndexSelector(arr.astype(np.intp, copy=False)).resolve_indices(dataset)
    raise ValueError(f"Unsupported selector type: {type(selector)}")


def _full_indices_to_masked_columns(
    mask: NDArray[np.bool_],
    full_indices: NDArray[np.intp],
    *,
    allow_empty: bool = False,
) -> NDArray[np.intp]:
    mask_indices = np.flatnonzero(mask)
    positions = np.searchsorted(mask_indices, full_indices)
    in_bounds = positions < mask_indices.size
    positions = positions[in_bounds]
    full_indices = full_indices[in_bounds]
    valid = mask_indices[positions] == full_indices
    selected = positions[valid].astype(np.intp, copy=False)
    skipped = int(len(full_indices) - np.count_nonzero(valid))
    if skipped:
        warnings.warn(
            f"{skipped} voxel(s) outside the dataset mask were ignored",
            stacklevel=2,
        )
    if selected.size == 0 and not allow_empty:
        raise ValueError("No requested voxels are within the dataset mask")
    return selected
