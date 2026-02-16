"""Mixed model solver for single-trial beta estimation.

Provides REML-based variance component estimation as an alternative
to LSS/OASIS.  Uses a simplified linear mixed model where trial
amplitudes are random effects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from ._project import project_nuisance
from ._types import SingleTrialResult


def _estimate_variance_components_reml(
    Y_resid: NDArray[np.float64],
    X_resid: NDArray[np.float64],
    method: str = "reml",
) -> tuple[float, float]:
    """Estimate variance components via simplified REML.

    For the model:
        Y = X @ u + e,  u ~ N(0, sigma2_u * I),  e ~ N(0, sigma2_e * I)

    Uses moment-based estimation from residuals.

    Parameters
    ----------
    Y_resid : (n, V)
        Residualised data (nuisance projected out).
    X_resid : (n, T)
        Residualised trial regressors.
    method : str
        "reml" or "ml" (currently both use moment-based estimation).

    Returns
    -------
    sigma2_u : float
        Trial effect variance.
    sigma2_e : float
        Residual variance.
    """
    n, T = X_resid.shape
    V = Y_resid.shape[1]

    # OLS estimates for each voxel
    XtX = X_resid.T @ X_resid  # (T, T)
    try:
        L = linalg.cho_factor(XtX)
        betas_ols = linalg.cho_solve(L, X_resid.T @ Y_resid)  # (T, V)
    except linalg.LinAlgError:
        betas_ols = np.linalg.lstsq(X_resid, Y_resid, rcond=None)[0]

    # Residual variance (across all voxels)
    Y_fitted = X_resid @ betas_ols  # (n, V)
    residuals = Y_resid - Y_fitted
    sse = float(np.sum(residuals ** 2))
    dof_resid = n * V - T * V
    sigma2_e = sse / max(dof_resid, 1.0)

    # Between-trial variance (from beta variance across voxels)
    # E[beta_j] = u_j, Var[beta_j] = sigma2_u + sigma2_e / ||x_j||^2
    # Average across trials and voxels
    diag_XtX = np.diag(XtX)
    diag_XtX_safe = np.maximum(diag_XtX, 1e-12)

    # Variance of betas
    beta_var = float(np.var(betas_ols, ddof=1))  # pooled across trials & voxels

    # Moment estimate: Var[beta] ≈ sigma2_u + mean(sigma2_e / ||x_j||^2)
    mean_x_norm = float(np.mean(diag_XtX_safe))
    sigma2_u = max(0.0, beta_var - sigma2_e / mean_x_norm)

    return sigma2_u, sigma2_e


def _blup_betas(
    Y_resid: NDArray[np.float64],
    X_resid: NDArray[np.float64],
    sigma2_u: float,
    sigma2_e: float,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute BLUP trial betas via ridge regression.

    The BLUP for random effects is:
        u_hat = (X'X + lambda * I)^{-1} X'Y
    where lambda = sigma2_e / sigma2_u.

    Parameters
    ----------
    Y_resid : (n, V)
    X_resid : (n, T)
    sigma2_u, sigma2_e : float
    eps : float

    Returns
    -------
    betas : (T, V)
    """
    n, T = X_resid.shape
    V = Y_resid.shape[1]

    # Ridge parameter
    if sigma2_u > eps:
        lambda_ridge = sigma2_e / sigma2_u
    else:
        # Fall back to OLS if random effect variance is zero
        lambda_ridge = 0.0

    # Regularised Gram matrix
    XtX = X_resid.T @ X_resid  # (T, T)
    G = XtX + lambda_ridge * np.eye(T, dtype=np.float64)
    XtY = X_resid.T @ Y_resid  # (T, V)

    # Solve via Cholesky if possible
    try:
        L = linalg.cho_factor(G)
        betas = linalg.cho_solve(L, XtY)
    except linalg.LinAlgError:
        betas = np.linalg.lstsq(G, XtY, rcond=None)[0]

    return betas


def mixed_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    trial_labels: Optional[list] = None,
    method: str = "reml",
) -> SingleTrialResult:
    """Estimate trial-wise betas via linear mixed model.

    Treats trial amplitudes as random effects and estimates them
    via REML/BLUP.  This provides shrinkage towards the population
    mean, which can improve estimates when trials have low SNR.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix.
    X : NDArray, shape ``(n, n_trials)``
        Trial regressor matrix.
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors (fixed effects).
    trial_labels : list of str, optional
    method : str
        ``"reml"`` (default) or ``"ml"``.

    Returns
    -------
    SingleTrialResult

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single import mixed_single_trial
    >>> n, n_trials, V = 100, 20, 50
    >>> Y = np.random.randn(n, V)
    >>> X = np.random.randn(n, n_trials)
    >>> result = mixed_single_trial(Y, X)
    >>> result.betas.shape
    (20, 50)
    >>> result.method
    'mixed'
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    n, n_trials = X.shape
    V = Y.shape[1]
    if Y.shape[0] != n:
        raise ValueError(
            f"Y has {Y.shape[0]} timepoints, X has {n}."
        )

    # Project out nuisance (fixed effects)
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        Y_clean, X_clean = project_nuisance(confounds, Y, X)
        nuis_rank = confounds.shape[1]
    else:
        Y_clean, X_clean = Y, X
        nuis_rank = 0

    # Estimate variance components
    sigma2_u, sigma2_e = _estimate_variance_components_reml(
        Y_clean, X_clean, method=method
    )

    # Compute BLUP (shrinkage estimates)
    betas = _blup_betas(Y_clean, X_clean, sigma2_u, sigma2_e)

    # Residual degrees of freedom
    # For mixed models: n*V - rank(X) - nuis_rank (approx)
    dof = max(1.0, float(n * V - n_trials * V - nuis_rank))

    # Store variance components in extra
    extra = {
        "sigma2_u": sigma2_u,
        "sigma2_e": sigma2_e,
        "lambda": sigma2_e / max(sigma2_u, 1e-12),
    }

    return SingleTrialResult(
        betas=betas,
        method="mixed",
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=dof,
        se=None,  # SE computation requires full covariance matrix
        extra=extra,
    )
