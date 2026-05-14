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
    import neuroim


def _spacing_from_affine(
    affine: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Pull voxel sizes from a 4x4 affine."""
    if affine.shape == (4, 4):
        scale = np.linalg.norm(affine[:3, :3], axis=0)
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    raise ValueError(f"Expected 4x4 affine; got shape {affine.shape}")


def _origin_from_affine(
    affine: NDArray[np.float64],
) -> tuple[float, float, float]:
    if affine.shape == (4, 4):
        return (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    raise ValueError(f"Expected 4x4 affine; got shape {affine.shape}")


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
        vec: NDArray,
        *,
        fill: float = np.nan,
        dtype: DTypeLike = np.float64,
    ) -> NDArray:
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

    def to_neuro_space(self) -> neuroim.NeuroSpace:
        """Build a 3-D ``neuroim.NeuroSpace`` for this context."""
        import neuroim  # type: ignore[import-untyped]

        spacing = self.spacing
        origin = self.origin
        if spacing is None or origin is None:
            if self.affine is not None:
                if spacing is None:
                    spacing = _spacing_from_affine(self.affine)
                if origin is None:
                    origin = _origin_from_affine(self.affine)
            else:
                spacing = spacing or (1.0, 1.0, 1.0)
                origin = origin or (0.0, 0.0, 0.0)

        return neuroim.NeuroSpace(
            dim=tuple(int(d) for d in self.spatial_shape),
            spacing=tuple(float(s) for s in spacing),
            origin=tuple(float(o) for o in origin),
        )

    def to_neurovol(
        self,
        vec: NDArray,
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
        vec: NDArray,
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
    def from_dataset(cls, dataset: object) -> SpatialContext | None:
        """Pull a :class:`SpatialContext` off a dataset / adapter, if possible.

        Returns ``None`` for non-spatial datasets (e.g. a bare matrix adapter
        without a 3-D mask).
        """
        if dataset is None:
            return None

        get_mask = getattr(dataset, "get_mask", None)
        if not callable(get_mask):
            return None
        try:
            mask_arr = np.asarray(get_mask(), dtype=bool)
        except Exception:
            return None
        if mask_arr.ndim != 3:
            return None

        affine: NDArray[np.float64] | None = None
        get_affine = getattr(dataset, "get_affine", None)
        if callable(get_affine):
            try:
                affine = np.asarray(get_affine(), dtype=np.float64)
            except Exception:
                affine = None

        # Pull spacing/origin from a NeuroVec source if available, since
        # that's the truthiest in the adapter; otherwise derive from affine.
        # FmriDataset wraps the adapter in ``_source``; peek through it.
        spacing: tuple[float, float, float] | None = None
        origin: tuple[float, float, float] | None = None
        adapter = dataset if hasattr(dataset, "_vecs") else getattr(dataset, "_source", None)
        vecs = getattr(adapter, "_vecs", None) if adapter is not None else None
        if vecs:
            space = getattr(vecs[0], "space", None)
            if space is not None and getattr(space, "ndim", 0) >= 3:
                try:
                    spacing_raw = tuple(float(v) for v in space.spacing[:3])
                    origin_raw = tuple(float(v) for v in space.origin[:3])
                    if len(spacing_raw) == 3 and len(origin_raw) == 3:
                        spacing = (
                            spacing_raw[0],
                            spacing_raw[1],
                            spacing_raw[2],
                        )
                        origin = (origin_raw[0], origin_raw[1], origin_raw[2])
                except Exception:
                    spacing = origin = None
        if affine is None and adapter is not None:
            inner_get_affine = getattr(adapter, "get_affine", None)
            if callable(inner_get_affine):
                try:
                    affine = np.asarray(inner_get_affine(), dtype=np.float64)
                except Exception:
                    affine = None

        return cls(
            mask=mask_arr,
            spatial_shape=(
                int(mask_arr.shape[0]),
                int(mask_arr.shape[1]),
                int(mask_arr.shape[2]),
            ),
            affine=affine,
            spacing=spacing,
            origin=origin,
        )

    @classmethod
    def from_model(cls, model: object) -> SpatialContext | None:
        """Pull a context off ``model.dataset`` if accessible."""
        if model is None:
            return None
        dataset = getattr(model, "dataset", None)
        return cls.from_dataset(dataset)
