"""Space descriptors for native group-level analysis.

The native Python model uses zero-based voxel indices internally. Readers for
R-origin ``/gds`` files should translate serialized one-based indices at the I/O
boundary instead of leaking that convention into the core space model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .dtypes import as_group_float_array, as_group_index_array
from .errors import GroupSpaceError

StorageMode = Literal["dense", "packed"]
Hemisphere = Literal["L", "R", "LR"]
AxisIndex = Union[Sequence[int], NDArray[Any]]


@runtime_checkable
class GroupSpace(Protocol):
    """Protocol shared by group-analysis space descriptors."""

    @property
    def kind(self) -> str:
        """Space kind identifier."""

    @property
    def n_samples(self) -> int:
        """Number of samples along the group-analysis sample axis."""

    def validate(self) -> None:
        """Validate descriptor consistency."""

    def subset(self, index: AxisIndex) -> GroupSpace:
        """Return a descriptor restricted to sample-axis indices."""


def _coerce_labels(labels: Sequence[object], *, parameter: str) -> tuple[str, ...]:
    out = tuple(str(label) for label in labels)
    if not out:
        raise GroupSpaceError(f"{parameter} must be non-empty")
    if any(label == "" for label in out):
        raise GroupSpaceError(f"{parameter} must not contain empty labels")
    if len(set(out)) != len(out):
        raise GroupSpaceError(f"{parameter} must be unique")
    return out


def _normalize_index(
    index: AxisIndex,
    *,
    length: int,
) -> NDArray[np.intp]:
    arr: NDArray[Any] = np.asarray(index)
    if arr.dtype == np.bool_:
        if arr.ndim != 1 or arr.shape[0] != length:
            raise GroupSpaceError("logical index length must match space samples")
        idx = np.flatnonzero(arr).astype(np.intp, copy=False)
    else:
        idx = as_group_index_array(arr).reshape(-1)

    if idx.size == 0:
        raise GroupSpaceError("subset index must be non-empty")
    if np.any(idx < 0) or np.any(idx >= length):
        raise GroupSpaceError("subset index contains values out of range")
    return idx


def _validate_spatial_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3:
        raise GroupSpaceError("voxel shape must have length 3")
    if any(x <= 0 for x in dims):
        raise GroupSpaceError("voxel shape entries must be positive")
    return dims


@dataclass(frozen=True)
class SampleLabelSpace:
    """Space descriptor for tabular samples such as ROIs or features."""

    labels: Sequence[object]
    kind: str = "sample_labels"

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _coerce_labels(self.labels, parameter="labels"))

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    def validate(self) -> None:
        _coerce_labels(self.labels, parameter="labels")

    def subset(self, index: AxisIndex) -> SampleLabelSpace:
        idx = _normalize_index(index, length=self.n_samples)
        return SampleLabelSpace([self.labels[int(i)] for i in idx])


@dataclass(frozen=True)
class ParcelSpace:
    """Space descriptor for parcel or ROI summaries."""

    labels: Sequence[object]
    lookup: Any = None
    membership: Any = None
    kind: str = "parcels"

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _coerce_labels(self.labels, parameter="labels"))

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    def validate(self) -> None:
        _coerce_labels(self.labels, parameter="labels")

    def subset(self, index: AxisIndex) -> ParcelSpace:
        idx = _normalize_index(index, length=self.n_samples)
        return ParcelSpace(
            labels=[self.labels[int(i)] for i in idx],
            lookup=self.lookup,
            membership=None,
        )


@dataclass(frozen=True)
class BasisSpace:
    """Space descriptor for latent or basis-component features."""

    n_components: int
    basis_name: str | None = None
    projector: Any = None
    voxel_space: VoxelSpace | None = None
    kind: str = "basis"

    def __post_init__(self) -> None:
        n_components = int(self.n_components)
        if n_components <= 0:
            raise GroupSpaceError("n_components must be positive")
        object.__setattr__(self, "n_components", n_components)

    @property
    def n_samples(self) -> int:
        return self.n_components

    def validate(self) -> None:
        if self.n_components <= 0:
            raise GroupSpaceError("n_components must be positive")
        if self.voxel_space is not None:
            self.voxel_space.validate()

    def subset(self, index: AxisIndex) -> BasisSpace:
        idx = _normalize_index(index, length=self.n_samples)
        return BasisSpace(
            n_components=int(idx.size),
            basis_name=self.basis_name,
            projector=None,
            voxel_space=self.voxel_space,
        )


@dataclass(frozen=True)
class SurfaceSpace:
    """Space descriptor for cortical surface vertices."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.intp]
    hemi: Hemisphere
    template_id: str | None = None
    kind: str = "surface"

    def __post_init__(self) -> None:
        vertices = as_group_float_array(self.vertices)
        faces = as_group_index_array(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise GroupSpaceError("vertices must be a 2-D array with 3 columns")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise GroupSpaceError("faces must be a 2-D array with 3 columns")
        if self.hemi not in ("L", "R", "LR"):
            raise GroupSpaceError("hemi must be one of: L, R, LR")
        if faces.size and (np.any(faces < 0) or np.any(faces >= vertices.shape[0])):
            raise GroupSpaceError("faces contain vertex indices out of range")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def n_samples(self) -> int:
        return int(self.vertices.shape[0])

    def validate(self) -> None:
        SurfaceSpace(
            vertices=self.vertices,
            faces=self.faces,
            hemi=self.hemi,
            template_id=self.template_id,
        )

    def subset(self, index: AxisIndex) -> SurfaceSpace:
        idx = _normalize_index(index, length=self.n_samples)
        remap = {int(old): new for new, old in enumerate(idx)}
        keep_faces = [
            [remap[int(v)] for v in face]
            for face in self.faces
            if all(int(v) in remap for v in face)
        ]
        faces = (
            np.asarray(keep_faces, dtype=np.intp)
            if keep_faces
            else np.empty((0, 3), dtype=np.intp)
        )
        return SurfaceSpace(
            vertices=self.vertices[idx],
            faces=faces,
            hemi=self.hemi,
            template_id=self.template_id,
        )


@dataclass(frozen=True)
class VoxelSpace:
    """Voxel-space descriptor with optional packed mask indices."""

    shape: Sequence[int]
    affine: NDArray[np.float64] | None = None
    mask_idx: Sequence[int] | NDArray[np.intp] | None = None
    storage: StorageMode = "dense"
    template_id: str | None = None
    kind: str = "voxel"

    def __post_init__(self) -> None:
        shape = _validate_spatial_shape(self.shape)
        affine = (
            np.eye(4, dtype=np.float64)
            if self.affine is None
            else as_group_float_array(self.affine)
        )
        if affine.shape != (4, 4):
            raise GroupSpaceError("affine must be a 4x4 matrix")
        if not np.all(np.isfinite(affine)):
            raise GroupSpaceError("affine must contain finite values")
        if self.storage not in ("dense", "packed"):
            raise GroupSpaceError("storage must be one of: dense, packed")

        mask_idx = None
        if self.mask_idx is not None:
            mask_idx = as_group_index_array(self.mask_idx).reshape(-1)
            n_voxels = int(np.prod(shape))
            if mask_idx.size == 0:
                raise GroupSpaceError("mask_idx must be non-empty when provided")
            if np.any(mask_idx < 0) or np.any(mask_idx >= n_voxels):
                raise GroupSpaceError("mask_idx contains indices out of range")
            if np.unique(mask_idx).size != mask_idx.size:
                raise GroupSpaceError("mask_idx must be unique")
        elif self.storage == "packed":
            raise GroupSpaceError("packed voxel spaces require mask_idx")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "mask_idx", mask_idx)

    @property
    def n_samples(self) -> int:
        if self.mask_idx is not None:
            return int(np.asarray(self.mask_idx).size)
        return int(np.prod(self.shape))

    def validate(self) -> None:
        VoxelSpace(
            shape=self.shape,
            affine=self.affine,
            mask_idx=self.mask_idx,
            storage=self.storage,
            template_id=self.template_id,
        )

    def subset(self, index: AxisIndex) -> VoxelSpace:
        idx = _normalize_index(index, length=self.n_samples)
        if self.mask_idx is not None:
            mask_idx = np.asarray(self.mask_idx, dtype=np.intp)[idx]
        else:
            mask_idx = idx
        return VoxelSpace(
            shape=self.shape,
            affine=self.affine,
            mask_idx=mask_idx,
            storage="packed",
            template_id=self.template_id,
        )


def assert_compatible_spaces(left: GroupSpace, right: GroupSpace) -> None:
    """Raise when two spaces cannot be compared along their sample axis."""
    left.validate()
    right.validate()
    if left.kind != right.kind:
        raise GroupSpaceError(f"space kinds differ: {left.kind!r} != {right.kind!r}")
    if isinstance(left, VoxelSpace) and isinstance(right, VoxelSpace):
        if left.shape != right.shape:
            raise GroupSpaceError("voxel shapes differ")
        if not np.allclose(np.asarray(left.affine), np.asarray(right.affine)):
            raise GroupSpaceError("voxel affines differ")
        return
    if left.n_samples != right.n_samples:
        raise GroupSpaceError("space sample counts differ")


def common_mask(
    left: VoxelSpace,
    right: VoxelSpace,
    *,
    rule: Literal["intersection", "union"] = "intersection",
) -> tuple[NDArray[np.intp], NDArray[np.intp], VoxelSpace]:
    """Return sample-axis indices and a common packed voxel space."""
    if rule not in ("intersection", "union"):
        raise GroupSpaceError("rule must be one of: intersection, union")
    assert_compatible_spaces(left, right)

    left_vox = (
        left.mask_idx
        if left.mask_idx is not None
        else np.arange(left.n_samples, dtype=np.intp)
    )
    right_vox = (
        right.mask_idx
        if right.mask_idx is not None
        else np.arange(right.n_samples, dtype=np.intp)
    )
    voxels = (
        np.intersect1d(left_vox, right_vox)
        if rule == "intersection"
        else np.union1d(left_vox, right_vox)
    )
    left_lookup = {int(v): i for i, v in enumerate(left_vox)}
    right_lookup = {int(v): i for i, v in enumerate(right_vox)}
    left_idx = np.asarray(
        [left_lookup[int(v)] for v in voxels if int(v) in left_lookup],
        dtype=np.intp,
    )
    right_idx = np.asarray(
        [right_lookup[int(v)] for v in voxels if int(v) in right_lookup],
        dtype=np.intp,
    )
    space = VoxelSpace(
        shape=left.shape,
        affine=left.affine,
        mask_idx=voxels,
        storage="packed",
        template_id=left.template_id,
    )
    return left_idx, right_idx, space
