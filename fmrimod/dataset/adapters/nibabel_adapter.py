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
    tr : float
        Repetition time in seconds.
    start_time : float
        Onset time of the first volume (default 0).

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
        tr: float = 2.0,
        start_time: float = 0.0,
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

        # Build or accept mask
        if mask is None:
            first_vol = self._images[0].get_fdata()[..., 0]
            self._mask = first_vol != 0
        elif isinstance(mask, np.ndarray):
            self._mask = mask.astype(bool)
        else:
            # Assume nibabel image
            self._mask = np.asarray(cast(Any, mask).get_fdata(), dtype=bool)

        self._tr = tr
        self._start_time = start_time
        self._n_voxels = int(self._mask.sum())

        # Cache time-point counts
        self._n_timepoints = [img.shape[-1] for img in self._images]

        # Affine from first image
        self._affine = self._images[0].affine

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Load run data as ``(time, voxels)``."""
        data_4d = self._images[run].get_fdata()
        # (nx, ny, nz, nt) → (nt, n_voxels)
        data_2d = data_4d[self._mask].T  # (nt, V)
        return np.asarray(data_2d, dtype=np.float64)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the 3-D boolean brain mask."""
        return cast("NDArray[np.bool_]", self._mask)

    def get_affine(self) -> NDArray[np.float64]:
        """Return the NIfTI affine matrix."""
        return cast("NDArray[np.float64]", self._affine)

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame for this dataset."""
        return SamplingFrame(
            blocklens=self._n_timepoints,
            tr=self._tr,
            start_time=self._start_time,
        )

    @property
    def n_runs(self) -> int:
        return len(self._images)

    @property
    def n_timepoints(self) -> List[int]:
        return self._n_timepoints

    @property
    def n_voxels(self) -> int:
        return self._n_voxels
