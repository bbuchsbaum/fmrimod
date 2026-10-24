"""NIfTI dataset adapter using nibabel.

Wraps one or more ``nibabel.Nifti1Image`` objects (one per run) into
a :class:`~fmrimod.dataset.protocols.DatasetProtocol`-compatible
adapter for use with the fmrimod GLM pipeline.

Requires ``nibabel`` (optional dependency).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ...sampling import SamplingFrame
from ..backends.nifti_backend import NiftiBackend
from ..errors import ConfigError


class NibabelAdapter:
    """Adapter wrapping nibabel NIfTI images as an fmrimod dataset.

    Parameters
    ----------
    images : Nifti1Image or sequence of Nifti1Image
        One image per run.  Each should be 4-D ``(nx, ny, nz, nt)``.
    mask : Nifti1Image or NDArray[bool], optional
        Brain mask.  If ``None``, all non-zero voxels in the first
        volume of the first run are used.
    tr : float or sequence of float
        Repetition time in seconds, or one value per run.
    start_time : float, sequence of float, or None
        Onset time of the first volume (default 0), or one value per run.
        ``None`` applies the :class:`SamplingFrame` default of ``TR/2`` per
        run.

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load("bold.nii.gz")
    >>> adapter = NibabelAdapter([img], tr=2.0)
    >>> adapter.get_data(0).shape  # (n_time, n_voxels)
    """

    def __init__(
        self,
        images: Sequence[Any],
        mask: Optional[object] = None,
        tr: float | Sequence[float] = 2.0,
        start_time: float | Sequence[float] | None = 0.0,
    ) -> None:
        try:
            __import__("nibabel")
        except ImportError as e:
            raise ConfigError(
                "nibabel is required for NibabelAdapter. "
                "Install with: pip install fmrimod[nibabel]",
                parameter="nibabel",
            ) from e

        # Normalise to list
        if not isinstance(images, (list, tuple)):
            images = [images]
        self._images = list(images)

        # Mask: the first volume's non-zero voxels, a boolean array, or an
        # image. Every run (and an image mask) must share run 0's grid and
        # affine; NiftiBackend checks that and reads volumes lazily.
        if mask is None:
            mask_source: Any = np.asarray(self._images[0].dataobj[..., 0]) != 0
        elif isinstance(mask, np.ndarray):
            mask_source = mask.astype(bool)
        else:
            mask_source = mask
        self.backend = NiftiBackend(self._images, mask_source)
        self.backend.open()

        self._n_timepoints = [int(img.shape[-1]) for img in self._images]
        self._sampling_frame = SamplingFrame(
            blocklens=self._n_timepoints, tr=tr, start_time=start_time
        )
        self._starts = np.concatenate([[0], np.cumsum(self._n_timepoints)])

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Load run data as ``(time, voxels)``."""
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"Run {run} out of range [0, {self.n_runs})")
        rows = np.arange(self._starts[run], self._starts[run + 1], dtype=np.intp)
        return self.backend.get_data(rows=rows)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the 3-D boolean brain mask."""
        return self.backend.space.mask_array()

    def get_affine(self) -> NDArray[np.float64]:
        """Return the NIfTI affine matrix."""
        return cast("NDArray[np.float64]", self.backend.space.affine)

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame for this dataset."""
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return len(self._images)

    @property
    def n_timepoints(self) -> List[int]:
        return self._n_timepoints

    @property
    def n_voxels(self) -> int:
        return int(self.backend.space.n_voxels)
