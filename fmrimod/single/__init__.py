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

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ._types import (
    OasisConfig,
    SbhmConfig,
    SingleTrialMethod,
    SingleTrialResult,
    VoxelHrfResult,
)
from ._prewhiten import PrewhitenConfig
from .lss import lss_single_trial
from .lsa import lsa_single_trial
from .oasis import oasis_single_trial
from .mixed import mixed_single_trial


def estimate_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    method: str = "lss",
    confounds: Optional[NDArray[np.float64]] = None,
    trial_labels: Optional[list] = None,
    return_se: bool = False,
    *,
    prewhiten: Optional[PrewhitenConfig] = None,
    oasis_config: Optional[OasisConfig] = None,
    sbhm_config: Optional[SbhmConfig] = None,
    sbhm_library: Optional[object] = None,
) -> SingleTrialResult:
    """Estimate per-trial betas using the specified method.

    This is the main dispatcher for single-trial estimation.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix (time x voxels).
    X : NDArray, shape ``(n, n_trials)`` or ``(n, n_trials * K)``
        Trial regressor matrix (already convolved with HRF).
    method : str
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
    oasis_config : OasisConfig, optional
        Configuration for the OASIS solver.
    sbhm_config : SbhmConfig, optional
        Configuration for the SBHM pipeline.
    sbhm_library : SbhmLibrary, optional
        Pre-built SBHM library (required for SBHM method).

    Returns
    -------
    SingleTrialResult
    """
    # Optional prewhitening
    if prewhiten is not None and prewhiten.method != "none":
        from ._prewhiten import prewhiten_matrices
        Y, X, confounds = prewhiten_matrices(Y, X, confounds, prewhiten)

    method_enum = SingleTrialMethod(method)

    if method_enum is SingleTrialMethod.LSS:
        return lss_single_trial(
            Y, X,
            confounds=confounds,
            return_se=return_se,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.LSA:
        return lsa_single_trial(
            Y, X,
            confounds=confounds,
            return_se=return_se,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.OASIS:
        cfg = oasis_config or OasisConfig(return_se=return_se)
        if return_se:
            cfg.return_se = True
        return oasis_single_trial(
            Y, X,
            confounds=confounds,
            config=cfg,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.SBHM:
        # Lazy import to avoid circular dependencies
        from .sbhm.pipeline import sbhm_single_trial
        cfg = sbhm_config or SbhmConfig()
        return sbhm_single_trial(
            Y, X,
            confounds=confounds,
            config=cfg,
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

    raise ValueError(f"Unknown method: {method!r}")


__all__ = [
    "estimate_single_trial",
    "lss_single_trial",
    "lsa_single_trial",
    "oasis_single_trial",
    "mixed_single_trial",
    "SingleTrialResult",
    "SingleTrialMethod",
    "OasisConfig",
    "SbhmConfig",
    "PrewhitenConfig",
    "VoxelHrfResult",
]
