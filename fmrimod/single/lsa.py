"""Least Squares All (LSA) estimation.

Standard GLM with all trial regressors entered simultaneously.
Serves as a baseline comparison for LSS and other methods.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..glm.solver import fast_preproject, fast_lm_matrix
from .lss import _build_adjustment_matrix
from ._types import SingleTrialMethod, SingleTrialResult


def lsa_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    return_se: bool = False,
    trial_labels: Optional[list] = None,
    baseline_regressors: Optional[NDArray[np.float64]] = None,
    include_intercept: bool = False,
) -> SingleTrialResult:
    """Estimate trial-wise betas via standard OLS (all trials simultaneous).

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix.
    X : NDArray, shape ``(n, n_trials)``
        Trial regressor matrix.
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors.
    return_se : bool
        If ``True``, compute standard errors.
    trial_labels : list of str, optional
        Labels for each trial.
    baseline_regressors : NDArray, shape ``(n, p)``, optional
        Baseline or experimental regressors included in the all-trials GLM.
    include_intercept : bool
        If ``True``, add an intercept to the adjustment design.

    Returns
    -------
    SingleTrialResult
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    n, n_trials = X.shape
    if Y.shape[0] != n:
        raise ValueError(
            f"Y has {Y.shape[0]} timepoints, X has {n}."
        )
    if trial_labels is not None and len(trial_labels) != n_trials:
        raise ValueError(
            f"trial_labels has length {len(trial_labels)}, expected {n_trials}."
        )

    # Build full design: [trials | confounds]
    adjustment = _build_adjustment_matrix(
        n,
        baseline_regressors=baseline_regressors,
        confounds=confounds,
        include_intercept=include_intercept,
    )
    if adjustment is not None:
        X_full = np.column_stack([X, adjustment])
    else:
        X_full = X

    proj = fast_preproject(X_full)
    result = fast_lm_matrix(X_full, Y, proj)

    # Extract trial betas only (first n_trials rows)
    betas = result.betas[:n_trials]  # (n_trials, V)

    se = None
    if return_se:
        # SE_j = sqrt(sigma2 * (X'X)^{-1}_{jj})
        diag_XtXinv = np.diag(proj.XtXinv)[:n_trials]  # (n_trials,)
        se = np.sqrt(
            result.sigma2[np.newaxis, :] * diag_XtXinv[:, np.newaxis]
        )

    return SingleTrialResult(
        betas=betas,
        method=SingleTrialMethod.LSA,
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=result.dfres,
        se=se,
    )
