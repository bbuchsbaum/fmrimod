"""Configuration dataclasses for fmri_lm fitting.

Ports R's ``fmri_lm_control()`` into Python dataclasses with validation.
"""

from __future__ import annotations

import warnings
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

    type: Union[Literal[False], Literal["FALSE"], Literal["huber"], Literal["bisquare"]] = False
    k_huber: float = 1.345
    c_tukey: float = 4.685
    max_iter: int = 2
    scale_scope: Literal["run", "global", "voxel", "local"] = "run"
    reestimate_phi: bool = False

    def __post_init__(self) -> None:
        try:
            self.k_huber = float(self.k_huber)
        except (TypeError, ValueError) as exc:
            raise ValueError("k_huber must be numeric") from exc
        try:
            self.c_tukey = float(self.c_tukey)
        except (TypeError, ValueError) as exc:
            raise ValueError("c_tukey must be numeric") from exc
        if not isinstance(self.max_iter, (int, np.integer)):
            raise ValueError("max_iter must be an integer >= 1")
        self.max_iter = int(self.max_iter)
        if self.type == "FALSE":
            self.type = False
        if self.type not in (False, "huber", "bisquare"):
            raise ValueError(
                f"robust type must be False, 'huber', or 'bisquare', got {self.type!r}"
            )
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.scale_scope == "local":
            self.scale_scope = "voxel"
        if self.scale_scope not in ("run", "global", "voxel"):
            raise ValueError("scale_scope must be 'run', 'global', or 'voxel'")
        if not isinstance(self.reestimate_phi, (bool, np.bool_)):
            raise ValueError("reestimate_phi must be a boolean scalar")

    @property
    def enabled(self) -> bool:
        return self.type is not False


@dataclass(init=False)
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

    def __init__(
        self,
        struct: Literal["iid", "ar1", "ar2", "arp", "arma"] = "iid",
        p: Optional[int] = None,
        iter_gls: int = 1,
        global_ar: bool = False,
        voxelwise: bool = False,
        exact_first: bool = False,
        censor: Optional[Union[NDArray, List[int], Literal["auto"]]] = None,
        method: Literal["ar", "arma", "afni"] = "ar",
        q: int = 0,
        p_max: int = 6,
        pooling: Literal["global", "run", "parcel"] = "global",
        convergence_tol: float = 5e-3,
        parcels: Optional[NDArray] = None,
        **kwargs,
    ) -> None:
        # R-compat alias: ar_options$global -> global_ar
        if "global" in kwargs:
            if global_ar is not False:
                raise TypeError("Specify only one of 'global' or 'global_ar'")
            global_ar = kwargs.pop("global")
        if kwargs:
            extras = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected AR option(s): {extras}")

        self.struct = struct
        self.p = p
        self.iter_gls = iter_gls
        self.global_ar = global_ar
        self.voxelwise = voxelwise
        self.exact_first = exact_first
        self.censor = censor
        self.method = method
        self.q = q
        self.p_max = p_max
        self.pooling = pooling
        self.convergence_tol = convergence_tol
        self.parcels = parcels
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.struct not in ("iid", "ar1", "ar2", "arp", "arma"):
            raise ValueError(
                "AR struct must be 'iid', 'ar1', 'ar2', 'arp', or 'arma'"
            )
        if self.struct == "arp" and self.p is None:
            raise ValueError("p must be specified when struct is 'arp'")
        if self.p is not None:
            if isinstance(self.p, (bool, np.bool_)) or not isinstance(
                self.p, (int, np.integer)
            ):
                raise ValueError("p must be an integer >= 1")
            self.p = int(self.p)
            if self.p < 1:
                raise ValueError("p must be >= 1")
        if isinstance(self.iter_gls, (bool, np.bool_)) or not isinstance(
            self.iter_gls, (int, np.integer)
        ):
            raise ValueError("iter_gls must be an integer >= 0")
        self.iter_gls = int(self.iter_gls)
        if self.iter_gls < 0:
            raise ValueError("iter_gls must be >= 0")
        if self.method not in ("ar", "arma", "afni"):
            raise ValueError("method must be 'ar', 'arma', or 'afni'")
        if self.pooling not in ("global", "run", "parcel"):
            raise ValueError("pooling must be 'global', 'run', or 'parcel'")
        if isinstance(self.q, (bool, np.bool_)) or not isinstance(
            self.q, (int, np.integer)
        ):
            raise ValueError("q must be an integer >= 0")
        self.q = int(self.q)
        if self.q < 0:
            raise ValueError("q must be >= 0")
        if isinstance(self.p_max, (bool, np.bool_)) or not isinstance(
            self.p_max, (int, np.integer)
        ):
            raise ValueError("p_max must be an integer >= 1")
        self.p_max = int(self.p_max)
        if self.p_max < 1:
            raise ValueError("p_max must be >= 1")
        if isinstance(self.convergence_tol, (bool, np.bool_)) or not isinstance(
            self.convergence_tol, (int, float, np.integer, np.floating)
        ):
            raise ValueError("convergence_tol must be numeric")
        self.convergence_tol = float(self.convergence_tol)
        if self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be > 0")
        for key, value in (
            ("global_ar", self.global_ar),
            ("voxelwise", self.voxelwise),
            ("exact_first", self.exact_first),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean scalar")
        if self.censor is not None:
            if isinstance(self.censor, str):
                if self.censor != "auto":
                    raise ValueError(
                        "censor must be None, 'auto', a numeric vector, or a logical vector"
                    )
            else:
                censor_arr = np.asarray(self.censor)
                if censor_arr.ndim == 0:
                    raise ValueError(
                        "censor must be None, 'auto', a numeric vector, or a logical vector"
                    )
                if censor_arr.dtype.kind not in ("b", "i", "u", "f"):
                    raise ValueError(
                        "censor must be None, 'auto', a numeric vector, or a logical vector"
                    )

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
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean scalar")
        if self.method not in ("inverse_squared", "soft_threshold", "tukey"):
            raise ValueError(
                "method must be one of: 'inverse_squared', 'soft_threshold', 'tukey'"
            )
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            if w.ndim != 1:
                raise ValueError("weights must be a 1-D array")
            if w.size == 0:
                raise ValueError("weights must contain at least one value")
            if not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            self.weights = w


@dataclass
class SoftSubspaceOptions:
    """Options for soft subspace projection (nuisance removal).

    Parameters
    ----------
    enabled : bool
        Whether to apply soft subspace projection.
    nuisance_matrix : NDArray or None
        Pre-computed nuisance time-series matrix.
    nuisance_mask : object or None
        Mask-style nuisance spec (R compatibility placeholder).
    lam : float or "auto" or "gcv"
        Regularisation parameter.
    warn_redundant : bool
        Warn if the baseline model already has nuisance terms.
    """

    enabled: bool = False
    nuisance_matrix: Optional[NDArray] = None
    nuisance_mask: Optional[object] = None
    lam: Union[float, Literal["auto"], Literal["gcv"]] = "auto"
    warn_redundant: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean scalar")
        if not isinstance(self.warn_redundant, (bool, np.bool_)):
            raise ValueError("warn_redundant must be a boolean scalar")
        if self.enabled and self.nuisance_matrix is None and self.nuisance_mask is None:
            raise ValueError(
                "soft subspace projection requires nuisance_matrix or nuisance_mask when enabled"
            )
        if isinstance(self.lam, str):
            if self.lam not in ("auto", "gcv"):
                raise ValueError("lambda must be a non-negative number, 'auto', or 'gcv'")
        elif isinstance(self.lam, (int, float)):
            if self.lam < 0:
                raise ValueError("lambda must be >= 0")
        else:
            raise ValueError("lambda must be a non-negative number, 'auto', or 'gcv'")


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
    solver : {"auto", "pinv"}
        Projection solver for OLS. ``"auto"`` keeps the fast default path;
        ``"pinv"`` uses a Moore-Penrose projection for strict reference
        parity workflows.
    """

    robust: RobustOptions = field(default_factory=RobustOptions)
    ar: AROptions = field(default_factory=AROptions)
    volume_weights: VolumeWeightOptions = field(default_factory=VolumeWeightOptions)
    soft_subspace: SoftSubspaceOptions = field(default_factory=SoftSubspaceOptions)
    solver: Literal["auto", "pinv"] = "auto"

    def __post_init__(self) -> None:
        if self.solver not in ("auto", "pinv"):
            raise ValueError("solver must be 'auto' or 'pinv'")

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
        if self.solver != "auto":
            parts.append(f"solver={self.solver}")
        if not parts:
            parts.append("OLS")
        return f"FmriLmConfig({', '.join(parts)})"


_LAMBDA_SENTINEL = object()


def soft_subspace_options(
    *,
    enabled: bool = False,
    nuisance_matrix: Optional[NDArray] = None,
    nuisance_mask: Optional[object] = None,
    lam: Union[float, Literal["auto", "gcv"], object] = _LAMBDA_SENTINEL,
    warn_redundant: bool = True,
    **kwargs,
) -> SoftSubspaceOptions:
    """Construct :class:`SoftSubspaceOptions` with R-style compatibility.

    Parameters mirror :class:`SoftSubspaceOptions` with the ``lambda`` alias
    accepted for R parity. A warning is emitted when both
    ``nuisance_matrix`` and ``nuisance_mask`` are supplied.
    """
    lam_value: Union[float, Literal["auto", "gcv"]] = "auto" if lam is _LAMBDA_SENTINEL else lam
    if kwargs:
        if "lambda" in kwargs:
            if lam is not _LAMBDA_SENTINEL:
                raise TypeError("Specify only one of 'lambda' or 'lam'")
            lam_value = kwargs.pop("lambda")
        if kwargs:
            extras = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected soft_subspace option(s): {extras}")

    if nuisance_matrix is not None and nuisance_mask is not None and warn_redundant:
        warnings.warn(
            "Both 'nuisance_matrix' and 'nuisance_mask' were provided.\n"
            "This is a redundant configuration and may be ignored depending on call path.",
            UserWarning,
            stacklevel=2,
        )

    return SoftSubspaceOptions(
        enabled=enabled,
        nuisance_matrix=nuisance_matrix,
        nuisance_mask=nuisance_mask,
        lam=lam_value,
        warn_redundant=warn_redundant,
    )


# Keep a stable local name for the factory after parameter shadowing in
# fmri_lm_control.
_soft_subspace_options = soft_subspace_options


def fmri_lm_control(
    robust_options: Optional[dict] = None,
    ar_options: Optional[dict] = None,
    volume_weights_options: Optional[dict] = None,
    soft_subspace_options: Optional[dict] = None,
    solver: Literal["auto", "pinv"] = "auto",
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
    ss_opts = dict(soft_subspace_options or {})
    ss = _soft_subspace_options(**ss_opts)
    return FmriLmConfig(
        robust=robust,
        ar=ar,
        volume_weights=vw,
        soft_subspace=ss,
        solver=solver,
    )
