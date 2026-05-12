"""Shared types for single-trial beta estimation.

Defines result containers and configuration dataclasses used across
all estimation methods (LSS, LSA, OASIS, SBHM, mixed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


class SingleTrialMethod(str, Enum):
    """Available single-trial estimation methods."""

    LSS = "lss"
    LSA = "lsa"
    OASIS = "oasis"
    SBHM = "sbhm"
    MIXED = "mixed"


@dataclass
class SingleTrialResult:
    """Result of single-trial beta estimation.

    Attributes
    ----------
    betas : NDArray, shape ``(n_trials, n_voxels)``
        Per-trial beta estimates.  For multi-basis methods (K > 1),
        shape is ``(n_trials * K, n_voxels)``.
    method : str
        Estimation method used (``"lss"``, ``"lsa"``, ``"oasis"``,
        ``"sbhm"``, ``"mixed"``).
    trial_labels : list of str or None
        Labels identifying each trial.
    residual_df : float
        Residual degrees of freedom.
    se : NDArray or None
        Standard errors, same shape as *betas*.  Only available when
        the method supports it and it was requested.
    extra : dict
        Method-specific extras (e.g. SBHM match info, OASIS diagnostics).
    """

    betas: NDArray[np.float64]
    method: str
    trial_labels: Optional[list] = None
    residual_df: float = 0.0
    se: Optional[NDArray[np.float64]] = None
    extra: dict = field(default_factory=dict)


@dataclass
class OasisConfig:
    """Configuration for the OASIS closed-form solver.

    Attributes
    ----------
    K : int
        Basis dimension (1 for single-HRF, >1 for multi-basis).
    ridge_mode : str
        ``"none"``, ``"fractional"``, or ``"absolute"``.
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
    ridge_mode: str = "none"
    ridge_x: float = 0.0
    ridge_b: float = 0.0
    return_se: bool = False
    block_cols: int = 4096

    def __post_init__(self) -> None:
        if int(self.K) != self.K or self.K < 1:
            raise ValueError("K must be a positive integer")
        self.K = int(self.K)
        if self.ridge_mode not in {"none", "fractional", "absolute"}:
            raise ValueError("ridge_mode must be one of: none, fractional, absolute")
        if self.ridge_x < 0:
            raise ValueError("ridge_x must be non-negative")
        if self.ridge_b < 0:
            raise ValueError("ridge_b must be non-negative")
        if int(self.block_cols) != self.block_cols or self.block_cols < 1:
            raise ValueError("block_cols must be a positive integer")
        self.block_cols = int(self.block_cols)


@dataclass
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
    amplitude_method : str
        ``"oasis_voxel"``, ``"global_ls"``, or ``"lss1"``.
    ridge_mode : str
        Ridge mode for amplitude estimation.
    ridge_lambda : float
        Ridge regularisation strength.
    """

    r: int = 3
    shrink: bool = True
    top_k: int = 1
    amplitude_method: str = "oasis_voxel"
    ridge_mode: str = "fractional"
    ridge_lambda: float = 0.02


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
    conditions: list
