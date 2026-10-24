"""Neuroim adapter: wraps ``neuroim.NeuroVec`` into the fmrimod dataset protocol.

This is the canonical NIfTI/IO path for fmrimod. It accepts one or more
:class:`neuroim.NeuroVec` instances (one per run) and exposes them as a
:class:`~fmrimod.dataset.protocols.DatasetProtocol`-compatible adapter.

The older :class:`~fmrimod.dataset.adapters.nibabel_adapter.NibabelAdapter`
remains as a compatibility shim for code that holds raw ``nibabel.Nifti1Image``
objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray

from ...sampling import SamplingFrame
from ..backends.concat_backend import ConcatBackend
from ..backends.neurovec_backend import (
    NeuroVecBackend,
    is_neurovec,
    is_neurovol,
    resolve_volume_mask,
)
from ..errors import ConfigError

if TYPE_CHECKING:
    import neuroim  # type: ignore[import-untyped]


_is_neurovec = is_neurovec
_is_neurovol = is_neurovol


class NeuroVecAdapter:
    """Adapt ``neuroim.NeuroVec`` instances to :class:`DatasetProtocol`.

    Parameters
    ----------
    vecs
        Single :class:`neuroim.NeuroVec` or sequence of NeuroVecs, one per run.
    mask
        Optional mask. Accepts a :class:`neuroim.LogicalNeuroVol`,
        :class:`neuroim.NeuroVol`, 3-D boolean ndarray, or path to a NIfTI
        volume. If ``None``, a default mask is built from non-zero voxels of
        the first volume of the first run.
    tr
        Repetition time in seconds, or one value per run.
    start_time
        Onset of the first volume (defaults to 0.0), or one value per run.
        ``None`` applies the :class:`SamplingFrame` default of ``TR/2`` per
        run.

    Examples
    --------
    >>> import neuroim as ni
    >>> from fmrimod.dataset.adapters import NeuroVecAdapter
    >>> vec = ni.read_vec("sub-01_bold.nii.gz")
    >>> adapter = NeuroVecAdapter(vec, tr=2.0)
    >>> adapter.get_data(0).shape
    (n_time, n_voxels)
    """

    def __init__(
        self,
        vecs: Union["neuroim.NeuroVec", Sequence["neuroim.NeuroVec"]],
        *,
        mask: object = None,
        tr: float | Sequence[float],
        start_time: float | Sequence[float] | None = 0.0,
    ) -> None:
        try:
            import neuroim  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ConfigError(
                "neuroim is required for NeuroVecAdapter. "
                "Install with: pip install neuroim",
                parameter="neuroim",
            ) from exc

        if _is_neurovec(vecs):
            self._vecs: List["neuroim.NeuroVec"] = [vecs]
        elif isinstance(vecs, (list, tuple)) and all(_is_neurovec(v) for v in vecs):
            if not vecs:
                raise ValueError("NeuroVecAdapter requires at least one NeuroVec")
            self._vecs = list(vecs)
        else:
            raise TypeError(
                "NeuroVecAdapter expects a NeuroVec or sequence of NeuroVec; "
                f"got {type(vecs).__name__}"
            )

        first = self._vecs[0]
        if first.space.ndim != 4:
            raise ValueError(
                f"NeuroVec must wrap a 4-D NeuroSpace; got {first.space.ndim}-D"
            )
        spatial_shape = tuple(int(d) for d in first.space.dim[:3])
        self._spatial_shape = spatial_shape

        for i, v in enumerate(self._vecs[1:], start=1):
            other = tuple(int(d) for d in v.space.dim[:3])
            if other != spatial_shape:
                raise ValueError(
                    f"Run {i} spatial shape {other} != run 0 spatial shape "
                    f"{spatial_shape}"
                )

        # One mask for every run; the runs' spaces must match run 0's
        # (shape and affine, fmridataset 7cb121e), which ConcatBackend checks.
        mask_arr = resolve_volume_mask(mask, first)
        self.backend = ConcatBackend([NeuroVecBackend(v, mask_arr) for v in self._vecs])
        self._n_timepoints = [int(v.space.dim[3]) for v in self._vecs]
        # Per-run TR and start_time pass straight through (R 99d19fe, #94).
        self._sampling_frame = SamplingFrame(
            blocklens=self._n_timepoints, tr=tr, start_time=start_time
        )
        self._starts = np.concatenate([[0], np.cumsum(self._n_timepoints)])

    # -- DatasetProtocol implementation --

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return ``(n_timepoints, n_voxels)`` data for one run."""
        if run < 0 or run >= len(self._vecs):
            raise IndexError(f"Run {run} out of range [0, {len(self._vecs)})")
        rows = np.arange(self._starts[run], self._starts[run + 1], dtype=np.intp)
        return self.backend.get_data(rows=rows)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the 3-D boolean mask."""
        return self.backend.space.mask_array()

    def get_affine(self) -> NDArray[np.float64]:
        """Return the 4x4 voxel-to-world affine shared by every run."""
        return cast("NDArray[np.float64]", self.backend.space.affine)

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame for all runs."""
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return len(self._vecs)

    @property
    def n_timepoints(self) -> List[int]:
        return list(self._n_timepoints)

    @property
    def n_voxels(self) -> int:
        return int(self.backend.space.n_voxels)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self._spatial_shape

    # -- Construction helpers --

    @classmethod
    def from_paths(
        cls,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        mask: object = None,
        tr: float | Sequence[float],
        start_time: float | Sequence[float] | None = 0.0,
    ) -> "NeuroVecAdapter":
        """Load NIfTI files via :func:`neuroim.read_vec` and wrap them."""
        try:
            import neuroim
        except ImportError as exc:
            raise ConfigError(
                "neuroim is required to load NIfTI paths. "
                "Install with: pip install neuroim",
                parameter="neuroim",
            ) from exc

        if isinstance(paths, (str, Path)):
            paths = [paths]
        vecs = [neuroim.read_vec(str(p)) for p in paths]
        if isinstance(mask, (str, Path)):
            mask = neuroim.read_vol(str(mask))
        return cls(vecs, mask=mask, tr=tr, start_time=start_time)

    @classmethod
    def from_array(
        cls,
        data: NDArray,
        *,
        spacing: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        mask: object = None,
        tr: float | Sequence[float],
        start_time: float | Sequence[float] | None = 0.0,
    ) -> "NeuroVecAdapter":
        """Wrap a 4-D ndarray as a single-run NeuroVecAdapter."""
        try:
            import neuroim
        except ImportError as exc:
            raise ConfigError(
                "neuroim is required to wrap 4-D ndarray input. "
                "Install with: pip install neuroim",
                parameter="neuroim",
            ) from exc

        arr = np.asarray(data)
        if arr.ndim != 4:
            raise ValueError(
                f"from_array expects a 4-D array (x, y, z, t); got {arr.ndim}-D"
            )
        spatial = tuple(int(d) for d in arr.shape[:3])
        spacing_arr = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        origin_arr = tuple(origin) if origin is not None else (0.0, 0.0, 0.0)
        # NeuroSpace requires spacing/origin to match the full dim count.
        spacing_full = tuple(spacing_arr) + (1.0,)
        origin_full = tuple(origin_arr) + (0.0,)
        space = neuroim.NeuroSpace(
            dim=spatial + (int(arr.shape[3]),),
            spacing=spacing_full,
            origin=origin_full,
        )
        vec = neuroim.DenseNeuroVec(arr.astype(np.float64), space)
        return cls(vec, mask=mask, tr=tr, start_time=start_time)
