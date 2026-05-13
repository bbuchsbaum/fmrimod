"""Shared types for single-trial beta estimation.

Defines result containers and configuration dataclasses used across
all estimation methods (LSS, LSA, OASIS, SBHM, mixed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


class SingleTrialMethod(str, Enum):
    """Available single-trial estimation methods."""

    LSS = "lss"
    LSA = "lsa"
    OASIS = "oasis"
    SBHM = "sbhm"
    MIXED = "mixed"
    LSS_VOXEL_HRF = "lss_voxel_hrf"


SingleTrialMethodLike = Union[SingleTrialMethod, Literal[
    "lss",
    "lsa",
    "oasis",
    "sbhm",
    "mixed",
    "lss_voxel_hrf",
]]
"""Public single-trial method selector accepted by the dispatcher."""


RidgeMode = Literal["none", "fractional", "absolute"]
"""Ridge regularisation mode shared across OASIS and SBHM configs."""

AmplitudeMethod = Literal["global_ls", "lss1", "oasis_voxel"]
"""Amplitude estimator used by the SBHM amplitude stage."""

_RIDGE_MODES: frozenset[str] = frozenset({"none", "fractional", "absolute"})
_AMPLITUDE_METHODS: frozenset[str] = frozenset({"global_ls", "lss1", "oasis_voxel"})


@dataclass(frozen=True)
class SpatialDescriptor:
    """Spatial-identity / capability label for a beta volume.

    Captures *where* a single-trial beta vector lives so downstream
    consumers can decide whether two results are spatially comparable
    without re-reading the source dataset. Populated only when
    :class:`SingleTrialResult` is produced from a typed
    :class:`~fmrimod.dataset.FmriDataset`; matrix-first callers of
    :func:`~fmrimod.single.estimate_single_trial` leave the result's
    ``spatial_descriptor`` field set to ``None``.

    Attributes
    ----------
    n_voxels : int
        Number of beta columns, i.e. ``betas.shape[1]``.
    mask_shape : tuple of int
        Shape of the spatial mask as returned by ``dataset.get_mask().shape``.
        For volumetric datasets this is the 3D grid shape; for flattened
        backends it is a 1-tuple.
    mask_n_true : int
        Number of in-mask voxels (``int(mask.sum())``). When equal to
        ``n_voxels`` the betas live in the in-mask subspace; a mismatch
        indicates the wrapper consumed all-voxel data rather than the
        masked subset.
    """

    n_voxels: int
    mask_shape: Tuple[int, ...]
    mask_n_true: int

    def __post_init__(self) -> None:
        if int(self.n_voxels) != self.n_voxels or self.n_voxels < 0:
            raise ValueError("n_voxels must be a non-negative integer")
        object.__setattr__(self, "n_voxels", int(self.n_voxels))
        object.__setattr__(self, "mask_shape", tuple(int(d) for d in self.mask_shape))
        if int(self.mask_n_true) != self.mask_n_true or self.mask_n_true < 0:
            raise ValueError("mask_n_true must be a non-negative integer")
        object.__setattr__(self, "mask_n_true", int(self.mask_n_true))


@dataclass
class SingleTrialResult:
    """Result of single-trial beta estimation.

    Attributes
    ----------
    betas : NDArray, shape ``(n_trials, n_voxels)``
        Per-trial beta estimates.  For multi-basis methods (K > 1),
        shape is ``(n_trials * K, n_voxels)``.
    method : SingleTrialMethod
        Estimation method used (``"lss"``, ``"lsa"``, ``"oasis"``,
        ``"sbhm"``, ``"mixed"``, ``"lss_voxel_hrf"``).
    trial_labels : list of str or None
        Labels identifying each trial.
    residual_df : float
        Residual degrees of freedom.
    se : NDArray or None
        Standard errors, same shape as *betas*.  Only available when
        the method supports it and it was requested.
    extra : dict
        Method-specific extras (e.g. SBHM match info, OASIS diagnostics).
    trial_table : pandas.DataFrame or None
        Per-trial event metadata aligned with ``betas`` rows. Populated
        only when the result was produced by
        :func:`~fmrimod.single.estimate_single_trial_from_dataset`; the
        matrix-first :func:`~fmrimod.single.estimate_single_trial` path
        leaves this ``None``.
    run_labels : tuple or None
        Per-trial run/block label, one entry per ``betas`` row. Sourced
        from the events ``run``/``block`` column when present. ``None``
        for matrix-first callers.
    subject_id : object or None
        Subject identifier carried by the dataset (``dataset.subject_id``
        or the singleton entry of ``dataset.subject_ids``). ``None`` when
        the dataset does not advertise one.
    spatial_descriptor : SpatialDescriptor or None
        Spatial-identity label describing the voxel space the betas live
        in. Populated only on the dataset path.
    """

    betas: NDArray[np.float64]
    method: SingleTrialMethod
    trial_labels: Optional[list[str]] = None
    residual_df: float = 0.0
    se: Optional[NDArray[np.float64]] = None
    extra: dict[str, Any] = field(default_factory=dict)
    trial_table: Optional["pd.DataFrame"] = None
    run_labels: Optional[Tuple[Any, ...]] = None
    subject_id: Optional[Any] = None
    spatial_descriptor: Optional[SpatialDescriptor] = None


@dataclass(frozen=True)
class OasisConfig:
    """Configuration for the OASIS closed-form solver.

    Attributes
    ----------
    K : int
        Basis dimension (1 for single-HRF, >1 for multi-basis).
    ridge_mode : RidgeMode
        ``"none"``, ``"fractional"``, or ``"absolute"`` (case-sensitive).
    ridge_x : float
        Ridge parameter for trial-specific regressors.
    ridge_b : float
        Ridge parameter for the aggregator regressor.
    return_se : bool
        Whether to compute standard errors.
    block_cols : int
        Voxel block size for memory-efficient processing.
    """

    K: int = 1
    ridge_mode: RidgeMode = "none"
    ridge_x: float = 0.0
    ridge_b: float = 0.0
    return_se: bool = False
    block_cols: int = 4096

    def __post_init__(self) -> None:
        if int(self.K) != self.K or self.K < 1:
            raise ValueError("K must be a positive integer")
        object.__setattr__(self, "K", int(self.K))
        if self.ridge_mode not in _RIDGE_MODES:
            raise ValueError("ridge_mode must be one of: none, fractional, absolute")
        if self.ridge_x < 0:
            raise ValueError("ridge_x must be non-negative")
        if self.ridge_b < 0:
            raise ValueError("ridge_b must be non-negative")
        if int(self.block_cols) != self.block_cols or self.block_cols < 1:
            raise ValueError("block_cols must be a positive integer")
        object.__setattr__(self, "block_cols", int(self.block_cols))


@dataclass(frozen=True)
class SbhmConfig:
    """Configuration for the SBHM pipeline.

    Attributes
    ----------
    r : int
        Number of SVD components to retain.
    shrink : bool
        Apply Stein shrinkage in matching step.
    top_k : int
        Number of library members to blend per voxel.
    amplitude_method : AmplitudeMethod
        ``"global_ls"``, ``"lss1"``, or ``"oasis_voxel"`` (case-sensitive).
    ridge_mode : RidgeMode
        ``"none"``, ``"fractional"``, or ``"absolute"`` (case-sensitive).
    ridge_lambda : float
        Ridge regularisation strength.
    """

    r: int = 3
    shrink: bool = True
    top_k: int = 1
    amplitude_method: AmplitudeMethod = "oasis_voxel"
    ridge_mode: RidgeMode = "fractional"
    ridge_lambda: float = 0.02

    def __post_init__(self) -> None:
        if int(self.r) != self.r or self.r < 1:
            raise ValueError("r must be a positive integer")
        object.__setattr__(self, "r", int(self.r))
        if int(self.top_k) != self.top_k or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        object.__setattr__(self, "top_k", int(self.top_k))
        if self.amplitude_method not in _AMPLITUDE_METHODS:
            raise ValueError(
                "amplitude_method must be one of: global_ls, lss1, oasis_voxel"
            )
        if self.ridge_mode not in _RIDGE_MODES:
            raise ValueError("ridge_mode must be one of: none, fractional, absolute")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")


@dataclass
class VoxelHrfResult:
    """Result of per-voxel HRF estimation.

    Attributes
    ----------
    coefficients : NDArray, shape ``(n_basis, n_voxels)``
        Basis coefficients per voxel.
    basis : object
        HRF basis object from ``fmrimod.hrf``.
    conditions : list of str
        Condition labels.
    """

    coefficients: NDArray[np.float64]
    basis: Any
    conditions: list[str]
