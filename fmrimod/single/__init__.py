"""Single-trial beta estimation for fMRI.

Provides multiple algorithms for estimating per-trial activation
amplitudes from event-related fMRI designs:

- **LSS** (Least Squares Separate): vectorized closed-form solver
- **LSA** (Least Squares All): standard OLS with all trials
- **OASIS**: optimised batched LSS with ridge regularisation
- **SBHM**: Shared-Basis HRF Matching pipeline
- **Mixed**: mixed-model solver

The primary entry point is :func:`estimate_single_trial`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from ._prewhiten import PrewhitenConfig
from ._project import NuisanceProjector, build_nuisance_projector
from ._types import (
    OasisConfig,
    SbhmConfig,
    SingleTrialMethod,
    SingleTrialMethodLike,
    SingleTrialResult,
    VoxelHrfResult,
)
from .item import (
    ItemBundle,
    ItemCovarianceBlockResult,
    ItemCovarianceMatrixResult,
    ItemCovarianceResult,
    ItemCvAggregate,
    ItemCvResult,
    ItemFoldMetrics,
    ItemFoldSplit,
    ItemPredictions,
    ItemWeightsResult,
    item_build_design,
    item_compute_u,
    item_cv,
    item_fit,
    item_from_lsa,
    item_predict,
    item_slice_fold,
)
from .lsa import lsa_single_trial
from .lss import lss_single_trial
from .mixed import mixed_single_trial
from .oasis import oasis_single_trial
from .voxel_hrf import estimate_hrf, estimate_voxel_hrf, lss_with_voxel_hrf

if TYPE_CHECKING:
    from .sbhm.library import SbhmLibrary


def estimate_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    method: SingleTrialMethodLike = "lss",
    confounds: NDArray[np.float64] | None = None,
    trial_labels: list[str] | None = None,
    return_se: bool = False,
    *,
    prewhiten: PrewhitenConfig | None = None,
    nuisance_projector: NuisanceProjector | None = None,
    chunk_size: int | None = None,
    oasis_config: OasisConfig | None = None,
    sbhm_config: SbhmConfig | None = None,
    sbhm_library: SbhmLibrary | None = None,
    baseline_regressors: NDArray[np.float64] | None = None,
    include_intercept: bool = False,
) -> SingleTrialResult:
    """Estimate per-trial betas using the specified method.

    This is the main dispatcher for single-trial estimation.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix (time x voxels).
    X : NDArray, shape ``(n, n_trials)`` or ``(n, n_trials * K)``
        Trial regressor matrix (already convolved with HRF).
    method : SingleTrialMethod or literal string
        One of ``"lss"``, ``"lsa"``, ``"oasis"``, ``"sbhm"``,
        ``"mixed"``.
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors (motion, drift, etc.).
    trial_labels : list of str, optional
        Labels identifying each trial.
    return_se : bool
        Whether to compute standard errors (not all methods support
        this).
    prewhiten : PrewhitenConfig, optional
        If provided, AR-based prewhitening is applied to Y, X, and
        confounds before estimation.
    nuisance_projector : NuisanceProjector, optional
        Precomputed nuisance projector for LSS. Useful when reusing the
        same confound matrix across repeated LSS fits.
    chunk_size : int, optional
        Voxel chunk size for LSS beta-only solves.
    oasis_config : OasisConfig, optional
        Configuration for the OASIS solver.
    sbhm_config : SbhmConfig, optional
        Configuration for the SBHM pipeline.
    sbhm_library : SbhmLibrary, optional
        Pre-built SBHM library (required for SBHM method).
    baseline_regressors : NDArray, shape ``(n, p)``, optional
        Baseline or experimental regressors included in every LSS model.
        Equivalent to the ``Z`` matrix in ``fmrilss::lss``.
    include_intercept : bool
        Add an intercept to the LSS adjustment design.

    Returns
    -------
    SingleTrialResult
    """
    # Optional prewhitening
    if prewhiten is not None and prewhiten.method != "none":
        from ._prewhiten import prewhiten_matrices
        Y, X, confounds = prewhiten_matrices(Y, X, confounds, prewhiten)

    try:
        method_enum = (
            method
            if isinstance(method, SingleTrialMethod)
            else SingleTrialMethod(method)
        )
    except ValueError as exc:
        valid = ", ".join(member.value for member in SingleTrialMethod)
        raise ValueError(f"method must be one of: {valid}") from exc
    if method_enum not in {SingleTrialMethod.LSS, SingleTrialMethod.LSA} and (
        baseline_regressors is not None or include_intercept
    ):
        raise ValueError(
            "baseline_regressors and include_intercept are currently supported "
            "only for method='lss' or method='lsa'."
        )

    if method_enum is SingleTrialMethod.LSS:
        return lss_single_trial(
            Y, X,
            confounds=confounds,
            nuisance_projector=nuisance_projector,
            chunk_size=chunk_size,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )

    if method_enum is SingleTrialMethod.LSA:
        return lsa_single_trial(
            Y, X,
            confounds=confounds,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )

    if method_enum is SingleTrialMethod.OASIS:
        oasis_cfg = oasis_config or OasisConfig(return_se=return_se)
        if return_se and not oasis_cfg.return_se:
            oasis_cfg = replace(oasis_cfg, return_se=True)
        return oasis_single_trial(
            Y, X,
            confounds=confounds,
            config=oasis_cfg,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.SBHM:
        # Lazy import to avoid circular dependencies
        from .sbhm.pipeline import sbhm_single_trial
        sbhm_cfg = sbhm_config or SbhmConfig()
        return sbhm_single_trial(
            Y, X,
            confounds=confounds,
            config=sbhm_cfg,
            trial_labels=trial_labels,
            library=sbhm_library,
        )

    if method_enum is SingleTrialMethod.MIXED:
        from .mixed import mixed_single_trial
        return mixed_single_trial(
            Y, X,
            confounds=confounds,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.LSS_VOXEL_HRF:
        raise ValueError(
            "method='lss_voxel_hrf' requires an estimated VoxelHrfResult; "
            "call lss_with_voxel_hrf(...) directly."
        )

    raise ValueError(f"Unknown method: {method!r}")


__all__ = [
    "estimate_single_trial",
    "lss_single_trial",
    "lsa_single_trial",
    "oasis_single_trial",
    "mixed_single_trial",
    "item_build_design",
    "item_compute_u",
    "item_fit",
    "item_predict",
    "item_slice_fold",
    "item_cv",
    "item_from_lsa",
    "estimate_hrf",
    "estimate_voxel_hrf",
    "lss_with_voxel_hrf",
    "SingleTrialResult",
    "SingleTrialMethod",
    "SingleTrialMethodLike",
    "OasisConfig",
    "SbhmConfig",
    "PrewhitenConfig",
    "VoxelHrfResult",
    "NuisanceProjector",
    "build_nuisance_projector",
    "ItemBundle",
    "ItemCovarianceResult",
    "ItemCovarianceMatrixResult",
    "ItemCovarianceBlockResult",
    "ItemWeightsResult",
    "ItemFoldSplit",
    "ItemFoldMetrics",
    "ItemPredictions",
    "ItemCvAggregate",
    "ItemCvResult",
]
