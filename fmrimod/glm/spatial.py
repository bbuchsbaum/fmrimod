"""Spatial reconstruction context for GLM outputs.

Carries just enough of the dataset's spatial metadata for an
``(n_voxels,)`` flat vector to be reconstructed into a 3-D volume and
exported as a :class:`neuroim.DenseNeuroVol` or NIfTI file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    import neuroim  # type: ignore[import-untyped]

    from fmrimod.dataset.space import VolumeSpace


@dataclass(frozen=True)
class SpatialContext:
    """Inverse-transform metadata for a masked voxel vector.

    Attributes
    ----------
    mask : NDArray[bool]
        3-D boolean mask aligned with the data the GLM was fit on.
        ``mask.sum() == n_voxels``.
    spatial_shape : tuple
        ``mask.shape``, repeated here for convenience.
    affine : NDArray[float] or None
        4x4 voxel-to-world affine. Optional; falls back to identity.
    spacing : tuple of float, optional
        Voxel spacing in mm. Inferred from ``affine`` when not given.
    origin : tuple of float, optional
        World-space origin in mm. Inferred from ``affine`` when not given.
    """

    mask: NDArray[np.bool_]
    spatial_shape: tuple[int, int, int]
    affine: NDArray[np.float64] | None = None
    spacing: tuple[float, float, float] | None = None
    origin: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.mask.shape != self.spatial_shape:
            raise ValueError(
                f"mask shape {self.mask.shape} != spatial_shape {self.spatial_shape}"
            )

    @property
    def n_voxels(self) -> int:
        return int(self.mask.sum())

    def reconstruct(
        self,
        vec: NDArray[np.float64],
        *,
        fill: float = np.nan,
        dtype: DTypeLike = np.float64,
    ) -> NDArray[np.float64]:
        """Inverse-mask a flat ``(n_voxels,)`` vector into a 3-D volume.

        Out-of-mask voxels are filled with ``fill`` (defaults to NaN).
        """
        vec = np.asarray(vec)
        if vec.ndim != 1:
            raise ValueError(f"reconstruct expects a 1-D vector; got {vec.ndim}-D")
        if vec.size != self.n_voxels:
            raise ValueError(
                f"reconstruct: vector length {vec.size} != n_voxels {self.n_voxels}"
            )
        out = np.full(self.spatial_shape, fill, dtype=dtype)
        out[self.mask] = vec
        return out

    def to_neuro_space(self) -> "neuroim.NeuroSpace":
        """Build a 3-D ``neuroim.NeuroSpace`` for this context."""
        import neuroim

        if self.affine is not None:
            return neuroim.NeuroSpace.from_affine(
                self.affine,
                self.spatial_shape,
            )

        spacing = self.spacing or (1.0, 1.0, 1.0)
        origin = self.origin or (0.0, 0.0, 0.0)

        return neuroim.NeuroSpace(
            dim=tuple(int(d) for d in self.spatial_shape),
            spacing=tuple(float(s) for s in spacing),
            origin=tuple(float(o) for o in origin),
        )

    def to_neurovol(
        self,
        vec: NDArray[np.float64],
        *,
        label: str = "",
        fill: float = 0.0,
    ) -> neuroim.DenseNeuroVol:
        """Build a :class:`neuroim.DenseNeuroVol` from a flat voxel vector.

        Non-mask voxels are filled with ``fill`` (defaults to 0.0 for clean
        NIfTI export; pass ``fill=np.nan`` for diagnostic visualization).
        """
        import neuroim

        volume = self.reconstruct(vec, fill=fill, dtype=np.float64)
        return neuroim.DenseNeuroVol(volume, self.to_neuro_space(), label=label)

    def write_nifti(
        self,
        vec: NDArray[np.float64],
        path: str | Path,
        *,
        label: str = "",
        fill: float = 0.0,
    ) -> Path:
        """Write a flat voxel vector to disk as a NIfTI volume."""
        import neuroim

        vol = self.to_neurovol(vec, label=label, fill=fill)
        out = Path(path)
        neuroim.write_vol(vol, str(out))
        return out

    # -- Construction --

    @classmethod
    def from_space(cls, space: VolumeSpace) -> SpatialContext | None:
        """Context for a dataset's :class:`~fmrimod.dataset.space.VolumeSpace`.

        ``None`` when the columns are not reported as a 3-D volume (matrix
        data).
        """
        if space.mask_shape is None or len(space.mask_shape) != 3:
            return None
        mask = space.mask_array()
        return cls(
            mask=mask,
            spatial_shape=(int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])),
            affine=None if space.affine is None else np.array(space.affine),
        )

    @classmethod
    def from_dataset(cls, dataset: object) -> SpatialContext | None:
        """Pull a :class:`SpatialContext` off a dataset or run-wise source.

        Returns ``None`` for non-spatial datasets (e.g. matrix data without a
        3-D mask).
        """
        if dataset is None:
            return None
        from fmrimod.dataset.space import VolumeSpace

        space = getattr(dataset, "space", None)
        if isinstance(space, VolumeSpace):
            return cls.from_space(space)
        backend = getattr(dataset, "backend", None)
        if isinstance(getattr(backend, "space", None), VolumeSpace):
            return cls.from_space(backend.space)

        # A user-written run-wise source: a 3-D mask and maybe an affine.
        get_mask = getattr(dataset, "get_mask", None)
        if not callable(get_mask):
            return None
        mask_arr = np.asarray(get_mask(), dtype=bool)
        if mask_arr.ndim != 3:
            return None
        affine: NDArray[np.float64] | None = None
        get_affine = getattr(dataset, "get_affine", None)
        if callable(get_affine):
            raw = get_affine()
            if raw is not None and np.shape(raw) == (4, 4):
                affine = np.asarray(raw, dtype=np.float64)
        return cls(
            mask=mask_arr,
            spatial_shape=(
                int(mask_arr.shape[0]),
                int(mask_arr.shape[1]),
                int(mask_arr.shape[2]),
            ),
            affine=affine,
        )

    @classmethod
    def from_model(cls, model: object) -> SpatialContext | None:
        """Pull a context off ``model.dataset`` if accessible."""
        if model is None:
            return None
        dataset = getattr(model, "dataset", None)
        return cls.from_dataset(dataset)
