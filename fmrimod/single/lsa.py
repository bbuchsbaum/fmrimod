"""Least Squares All (LSA) estimation.

Standard GLM with all trial regressors entered simultaneously.
Serves as a baseline comparison for LSS and other methods.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..glm.solver import fast_preproject, fast_lm_matrix
from ._types import SingleTrialResult


def lsa_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    return_se: bool = False,
    trial_labels: Optional[list] = None,
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

    # Build full design: [trials | confounds]
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        X_full = np.column_stack([X, confounds])
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
        method="lsa",
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=result.dfres,
        se=se,
    )
