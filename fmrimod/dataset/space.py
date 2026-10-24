"""Spatial identity of a dataset's columns.

A dataset's data matrix is ``time x in-mask voxels``. :class:`VolumeSpace`
says which voxels those columns are: the grid they live on, the in-mask
voxels in column order, and the voxel-to-world affine when it is known.
Two sources are spatially compatible only when their grids and affines
agree, never because their shapes happen to match (fmridataset ADR-001,
``7cb121e``).

neuroim-python is an optional extra, so this type is native; it converts
to and from :class:`neuroim.NeuroSpace` at the IO boundary and to the group
layer's :class:`~fmrimod.group.VoxelSpace` at the study hand-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fmrimod.group import VoxelSpace

_AFFINE_ATOL = 1e-6


@dataclass(frozen=True, eq=False)
class VolumeSpace:
    """Grid, in-mask voxels and affine of a dataset's columns.

    Parameters
    ----------
    shape
        Spatial grid ``(x, y, z)``. A non-spatial matrix uses ``(n, 1, 1)``.
    mask
        Boolean mask over the grid, flat in C order (length ``prod(shape)``)
        or in the grid's shape. Column ``j`` of the data is the ``j``-th
        ``True`` voxel in C order.
    affine
        ``4 x 4`` voxel-to-world transform, or ``None`` when the source has
        no geometry (matrix data, parcel features).
    mask_shape
        Shape in which :meth:`mask_array` reports the mask: the grid for
        image sources, ``None`` for a flat mask (matrix sources), or the
        shape a caller gave an in-memory mask in.
    """

    shape: tuple[int, int, int]
    mask: NDArray[np.bool_]
    affine: NDArray[np.float64] | None = None
    mask_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.shape)
        if len(shape) != 3 or any(v < 1 for v in shape):
            raise ValueError(f"shape must be three positive sizes; got {self.shape}")
        mask = np.asarray(self.mask)
        if mask.dtype != np.bool_:
            raise ValueError("mask must be boolean")
        if mask.size != int(np.prod(shape)) or mask.ndim not in (1, 3):
            raise ValueError(
                f"mask of shape {mask.shape} does not cover grid {shape}"
            )
        if mask.ndim == 3 and mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} != grid {shape}")
        flat = np.array(mask.reshape(-1), dtype=np.bool_)
        flat.setflags(write=False)
        affine = self.affine
        if affine is not None:
            affine = np.array(affine, dtype=np.float64)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise ValueError("affine must be a finite 4x4 matrix")
            affine.setflags(write=False)
        mask_shape = self.mask_shape
        if mask_shape is not None:
            mask_shape = tuple(int(v) for v in mask_shape)
            if int(np.prod(mask_shape)) != flat.size:
                raise ValueError(f"mask_shape {mask_shape} does not hold {flat.size} cells")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "mask", flat)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "mask_shape", mask_shape)

    @classmethod
    def from_neurospace(cls, space: Any, mask: NDArray[np.bool_]) -> VolumeSpace:
        """Build from a 3- or 4-D :class:`neuroim.NeuroSpace` and a mask."""
        shape = tuple(int(d) for d in space.dim[:3])
        return cls(shape=shape, mask=np.asarray(mask, dtype=bool),  # type: ignore[arg-type]
                   affine=np.asarray(space.affine, dtype=np.float64), mask_shape=shape)

    @property
    def n_voxels(self) -> int:
        """Number of in-mask voxels, i.e. data columns."""
        return int(np.count_nonzero(self.mask))

    @property
    def voxel_ids(self) -> NDArray[np.intp]:
        """C-order linear grid index of each column."""
        return np.flatnonzero(self.mask).astype(np.intp)

    def mask_array(self) -> NDArray[np.bool_]:
        """The mask in :attr:`mask_shape` (flat when ``None``), as a copy."""
        flat = self.mask.copy()
        return flat if self.mask_shape is None else flat.reshape(self.mask_shape)

    def same_grid(self, other: VolumeSpace) -> bool:
        """True when both spaces share the grid and the world transform.

        An unknown affine only matches another unknown affine, so geometry
        is never assumed.
        """
        if self.shape != other.shape:
            return False
        if self.affine is None or other.affine is None:
            return self.affine is None and other.affine is None
        return bool(np.allclose(self.affine, other.affine, atol=_AFFINE_ATOL))

    def require_same_grid(self, other: VolumeSpace, what: str) -> None:
        """Raise ``ValueError`` naming ``what`` unless it is on this grid."""
        if self.shape != other.shape:
            raise ValueError(f"{what} grid {other.shape} != {self.shape}")
        if not self.same_grid(other):
            raise ValueError(
                f"{what} is not in the same world space: affine "
                f"{None if other.affine is None else other.affine.tolist()} != "
                f"{None if self.affine is None else self.affine.tolist()}"
            )

    def require_same_columns(self, other: VolumeSpace, what: str) -> None:
        """Raise unless ``other`` has this grid, affine and in-mask voxels."""
        self.require_same_grid(other, what)
        if not np.array_equal(self.mask, other.mask):
            raise ValueError(f"{what} has a different mask")

    def to_voxel_space(self) -> VoxelSpace:
        """Packed group-layer :class:`~fmrimod.group.VoxelSpace` of the columns."""
        from fmrimod.group import VoxelSpace

        return VoxelSpace(
            shape=self.shape,
            affine=self.affine,
            mask_idx=self.voxel_ids,
            storage="packed",
        )

    def __repr__(self) -> str:
        geometry = "no affine" if self.affine is None else "affine"
        return f"VolumeSpace(shape={self.shape}, n_voxels={self.n_voxels}, {geometry})"


__all__ = ["VolumeSpace"]
