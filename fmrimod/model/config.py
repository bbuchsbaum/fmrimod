"""Configuration dataclasses for fmri_lm fitting.

Ports R's ``fmri_lm_control()`` into Python dataclasses with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class RobustOptions:
    """Options for robust (IRLS) fitting.

    Parameters
    ----------
    type : False or "huber" or "bisquare"
        Type of robust estimator.  ``False`` disables robust fitting.
    k_huber : float
        Tuning constant for Huber weights.
    c_tukey : float
        Tuning constant for Tukey bisquare weights.
    max_iter : int
        Maximum IRLS iterations.
    scale_scope : str
        How to estimate residual scale: ``"run"`` (pooled within run),
        ``"global"`` (pooled across all runs), or ``"voxel"`` (per-voxel).
    reestimate_phi : bool
        Whether to re-estimate AR parameters after robust weighting.
    """

    type: Union[Literal[False], Literal["huber"], Literal["bisquare"]] = False
    k_huber: float = 1.345
    c_tukey: float = 4.685
    max_iter: int = 2
    scale_scope: Literal["run", "global", "voxel"] = "run"
    reestimate_phi: bool = False

    def __post_init__(self) -> None:
        if self.type not in (False, "huber", "bisquare"):
            raise ValueError(
                f"robust type must be False, 'huber', or 'bisquare', got {self.type!r}"
            )
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.scale_scope not in ("run", "global", "voxel"):
            raise ValueError("scale_scope must be 'run', 'global', or 'voxel'")

    @property
    def enabled(self) -> bool:
        return self.type is not False


@dataclass
class AROptions:
    """Options for autoregressive noise modelling.

    Parameters
    ----------
    struct : str
        AR structure: ``"iid"`` (no AR), ``"ar1"``, ``"ar2"``, ``"arp"``.
    p : int or None
        AR order when *struct* is ``"arp"``.
    iter_gls : int
        Number of GLS iterations.
    global_ar : bool
        If ``True``, estimate a single global AR parameter set.
    voxelwise : bool
        If ``True``, estimate AR parameters per voxel.
    exact_first : bool
        If ``True``, use exact likelihood for the first *p* observations.
    censor : array-like or "auto" or None
        Timepoints to exclude from AR estimation.
    """

    struct: Literal["iid", "ar1", "ar2", "arp", "arma"] = "iid"
    p: Optional[int] = None
    iter_gls: int = 1
    global_ar: bool = False
    voxelwise: bool = False
    exact_first: bool = False
    censor: Optional[Union[NDArray, List[int], Literal["auto"]]] = None
    method: Literal["ar", "arma", "afni"] = "ar"
    q: int = 0
    p_max: int = 6
    pooling: Literal["global", "run", "parcel"] = "global"
    convergence_tol: float = 5e-3
    parcels: Optional[NDArray] = None

    def __post_init__(self) -> None:
        if self.struct not in ("iid", "ar1", "ar2", "arp", "arma"):
            raise ValueError(
                "AR struct must be 'iid', 'ar1', 'ar2', 'arp', or 'arma'"
            )
        if self.struct == "arp" and self.p is None:
            raise ValueError("p must be specified when struct is 'arp'")
        if self.p is not None and self.p < 1:
            raise ValueError("p must be >= 1")
        if self.iter_gls < 0:
            raise ValueError("iter_gls must be >= 0")

    @property
    def ar_order(self) -> int:
        """Effective AR order."""
        if self.struct == "iid":
            return 0
        elif self.struct == "ar1":
            return 1
        elif self.struct == "ar2":
            return 2
        elif self.struct == "arp":
            return self.p  # type: ignore[return-value]
        elif self.struct == "arma":
            return self.p if self.p is not None else 1
        return 0

    @property
    def enabled(self) -> bool:
        return self.struct != "iid"


@dataclass
class VolumeWeightOptions:
    """Options for volume (scan) weighting.

    Parameters
    ----------
    enabled : bool
        Whether to apply volume weights.
    method : str
        Weighting method: ``"inverse_squared"``, ``"soft_threshold"``,
        or ``"tukey"``.
    threshold : float
        DVARS threshold for weighting.
    weights : NDArray or None
        Pre-computed weight vector (overrides method/threshold).
    """

    enabled: bool = False
    method: Literal["inverse_squared", "soft_threshold", "tukey"] = "inverse_squared"
    threshold: float = 1.5
    weights: Optional[NDArray] = None

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")


@dataclass
class SoftSubspaceOptions:
    """Options for soft subspace projection (nuisance removal).

    Parameters
    ----------
    enabled : bool
        Whether to apply soft subspace projection.
    nuisance_matrix : NDArray or None
        Pre-computed nuisance time-series matrix.
    lam : float or "auto" or "gcv"
        Regularisation parameter.
    warn_redundant : bool
        Warn if the baseline model already has nuisance terms.
    """

    enabled: bool = False
    nuisance_matrix: Optional[NDArray] = None
    lam: Union[float, Literal["auto"], Literal["gcv"]] = "auto"
    warn_redundant: bool = True

    def __post_init__(self) -> None:
        if self.enabled and self.nuisance_matrix is None:
            raise ValueError(
                "nuisance_matrix is required when soft subspace projection is enabled"
            )
        if isinstance(self.lam, (int, float)) and self.lam < 0:
            raise ValueError("lambda must be >= 0")


@dataclass
class FmriLmConfig:
    """Complete configuration for GLM fitting.

    Collects all sub-option groups into one object, analogous to R's
    ``fmri_lm_control()``.

    Parameters
    ----------
    robust : RobustOptions
        Robust fitting options.
    ar : AROptions
        AR noise modelling options.
    volume_weights : VolumeWeightOptions
        Volume weighting options.
    soft_subspace : SoftSubspaceOptions
        Soft subspace projection options.
    """

    robust: RobustOptions = field(default_factory=RobustOptions)
    ar: AROptions = field(default_factory=AROptions)
    volume_weights: VolumeWeightOptions = field(default_factory=VolumeWeightOptions)
    soft_subspace: SoftSubspaceOptions = field(default_factory=SoftSubspaceOptions)

    def __repr__(self) -> str:
        parts = []
        if self.robust.enabled:
            parts.append(f"robust={self.robust.type}")
        if self.ar.enabled:
            parts.append(f"ar={self.ar.struct}")
        if self.volume_weights.enabled:
            parts.append(f"vol_weights={self.volume_weights.method}")
        if self.soft_subspace.enabled:
            parts.append("soft_subspace=True")
        if not parts:
            parts.append("OLS")
        return f"FmriLmConfig({', '.join(parts)})"


def fmri_lm_control(
    robust_options: Optional[dict] = None,
    ar_options: Optional[dict] = None,
    volume_weights_options: Optional[dict] = None,
    soft_subspace_options: Optional[dict] = None,
) -> FmriLmConfig:
    """Create an :class:`FmriLmConfig` from dictionaries.

    This is a convenience function mirroring R's ``fmri_lm_control()``.
    Keyword arguments in each dict override the defaults.

    Parameters
    ----------
    robust_options : dict, optional
        Overrides for :class:`RobustOptions`.
    ar_options : dict, optional
        Overrides for :class:`AROptions`.
    volume_weights_options : dict, optional
        Overrides for :class:`VolumeWeightOptions`.
    soft_subspace_options : dict, optional
        Overrides for :class:`SoftSubspaceOptions`.

    Returns
    -------
    FmriLmConfig
    """
    robust = RobustOptions(**(robust_options or {}))
    ar = AROptions(**(ar_options or {}))
    vw = VolumeWeightOptions(**(volume_weights_options or {}))
    ss = SoftSubspaceOptions(**(soft_subspace_options or {}))
    return FmriLmConfig(robust=robust, ar=ar, volume_weights=vw, soft_subspace=ss)
