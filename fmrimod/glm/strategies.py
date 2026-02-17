"""Fitting strategies for fMRI GLMs.

Implements runwise and chunkwise strategies for processing multi-run
fMRI datasets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .solver import Projection, fast_preproject, fast_lm_matrix, LmResult
from .preprocess import (
    apply_censoring,
    apply_volume_weights,
    compute_dvars,
    dvars_weights,
    soft_subspace_projection,
)
from ..model.config import FmriLmConfig


def fit_run_ols(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: FmriLmConfig,
    censor: Optional[NDArray[np.bool_]] = None,
) -> Tuple[LmResult, Projection, NDArray[np.float64], NDArray[np.float64]]:
    """Fit a single run with OLS (possibly after preprocessing).

    Parameters
    ----------
    X : NDArray
        Design matrix for this run, shape ``(n, p)``.
    Y : NDArray
        Data matrix for this run, shape ``(n, V)``.
    config : FmriLmConfig
        Fitting configuration.
    censor : NDArray[bool], optional
        Censoring vector for this run.

    Returns
    -------
    result : LmResult
        Regression result.
    proj : Projection
        Pre-computed projection (for later contrast computation).
    X_used : NDArray
        The (possibly preprocessed) design matrix that was actually fitted.
    Y_used : NDArray
        The (possibly preprocessed) data matrix that was actually fitted.
    """
    X_fit = X.copy()
    Y_fit = Y.copy()

    # 1. Censoring — coerce integer 0/1 to boolean for fmrireg parity
    if censor is not None:
        censor = np.asarray(censor)
        if censor.dtype.kind in ("i", "u", "f"):
            unique = np.unique(censor)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError("Censor vector must be boolean or binary (0/1)")
            censor = censor.astype(bool)
    if censor is not None and np.any(censor):
        X_fit, Y_fit, _ = apply_censoring(X_fit, Y_fit, censor)

    # 2. Volume weighting
    if config.volume_weights.enabled:
        if config.volume_weights.weights is not None:
            weights = config.volume_weights.weights
            if censor is not None and np.any(censor):
                n_rows = X.shape[0]
                n_keep = X_fit.shape[0]
                n_weights = weights.shape[0]
                if n_weights == n_rows:
                    weights = weights[~censor]
                elif n_weights != n_keep:
                    raise ValueError(
                        f"weights length {n_weights} does not match number of timepoints "
                        f"{n_rows} or uncensored rows {n_keep}"
                    )
        else:
            dvars = compute_dvars(Y_fit)
            weights = dvars_weights(
                dvars,
                method=config.volume_weights.method,
                threshold=config.volume_weights.threshold,
            )
        X_fit, Y_fit = apply_volume_weights(X_fit, Y_fit, weights)

    # 3. Soft subspace projection
    if config.soft_subspace.enabled and config.soft_subspace.nuisance_matrix is not None:
        nuisance = config.soft_subspace.nuisance_matrix
        if censor is not None and np.any(censor):
            nuisance = nuisance[~censor]
        lam_val = config.soft_subspace.lam
        if isinstance(lam_val, str):
            raise NotImplementedError(
                f"soft subspace lam='{lam_val}' is not implemented yet"
            )
        X_fit, Y_fit = soft_subspace_projection(X_fit, Y_fit, nuisance, lam_val)

    # 4. Fit OLS
    proj = fast_preproject(X_fit)
    result = fast_lm_matrix(X_fit, Y_fit, proj, return_fitted=True)

    return result, proj, X_fit, Y_fit


def fit_runwise(
    model: object,  # FmriModel
    config: FmriLmConfig,
) -> Dict:
    """Fit the GLM run by run and pool results.

    This is the default strategy: each run is fitted independently,
    then beta estimates and variances are combined via fixed-effects
    meta-analysis (inverse-variance weighting).

    Parameters
    ----------
    model : FmriModel
        The fMRI model (provides design matrices and data).
    config : FmriLmConfig
        Fitting configuration.

    Returns
    -------
    dict
        Dictionary with keys ``betas``, ``sigma``, ``dfres``, ``XtXinv``,
        ``projections``, ``run_results``, ``residuals``.
    """
    n_runs = model.n_runs  # type: ignore[attr-defined]

    run_results: List[LmResult] = []
    run_projections: List[Projection] = []
    run_residuals: List[NDArray] = []
    run_X: List[NDArray] = []

    for r in range(n_runs):
        # Get per-run data and design
        Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
        X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

        # Get censor for this run
        censor_r = None
        dataset = model.dataset  # type: ignore[attr-defined]
        if hasattr(dataset, "get_censor"):
            censor_r = dataset.get_censor(r)

        result, proj, X_used, Y_used = fit_run_ols(X_r, Y_r, config, censor_r)
        run_results.append(result)
        run_projections.append(proj)
        run_X.append(X_used)

        # Compute residuals
        if result.fitted is not None:
            run_residuals.append(Y_used - result.fitted)
        else:
            run_residuals.append(Y_used - X_used @ result.betas)

    # Pool across runs via fixed-effects meta-analysis
    pooled = _pool_run_results(run_results, run_projections)

    return {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": run_projections,
        "run_results": run_results,
        "residuals": run_residuals,
        "run_X": run_X,
    }


def _pool_run_results(
    results: List[LmResult],
    projections: List[Projection],
) -> Dict:
    """Pool per-run OLS results via fixed-effects meta-analysis.

    Uses inverse-variance weighting to combine betas across runs.

    Parameters
    ----------
    results : list of LmResult
        Per-run regression results.
    projections : list of Projection
        Per-run projections.

    Returns
    -------
    dict
        Pooled ``betas``, ``sigma``, ``dfres``, ``XtXinv``.
    """
    if len(results) == 1:
        r = results[0]
        p = projections[0]
        return {
            "betas": r.betas,
            "sigma": np.sqrt(r.sigma2),
            "dfres": r.dfres,
            "XtXinv": p.XtXinv,
        }

    p_dim = results[0].betas.shape[0]
    V = results[0].betas.shape[1]

    # Inverse-variance beta pooling (fmrireg parity).
    # se_{r,j,v}^2 = sigma2_{r,v} * XtXinv_{r,j,j}
    beta_stack = np.stack([r.betas for r in results], axis=0)  # (R, p, V)
    sigma2_stack = np.stack([r.sigma2 for r in results], axis=0)  # (R, V)
    diag_xtxinv = np.stack(
        [np.maximum(np.diag(proj.XtXinv), 0.0) for proj in projections], axis=0
    )  # (R, p)
    se2 = diag_xtxinv[:, :, np.newaxis] * sigma2_stack[:, np.newaxis, :]  # (R, p, V)
    weights = np.where(se2 > np.finfo(np.float64).eps, 1.0 / se2, 0.0)
    wsum = np.sum(weights, axis=0)  # (p, V)
    wbeta = np.sum(weights * beta_stack, axis=0)
    betas_pooled = np.divide(
        wbeta,
        wsum,
        out=np.mean(beta_stack, axis=0),
        where=wsum > 0.0,
    )

    # Keep a pooled XtXinv for downstream contrasts/SE interfaces.
    # This follows the prior behavior based on summed per-run information.
    XtX_total = np.zeros((p_dim, p_dim))
    rss_total = np.zeros(V)
    dfres_total = 0.0

    for r, proj in zip(results, projections):
        try:
            XtX_r = np.linalg.inv(proj.XtXinv)
        except np.linalg.LinAlgError:
            XtX_r = np.linalg.pinv(proj.XtXinv)

        XtX_total += XtX_r
        rss_total += r.rss
        dfres_total += r.dfres

    try:
        XtXinv_total = np.linalg.inv(XtX_total)
    except np.linalg.LinAlgError:
        XtXinv_total = np.linalg.pinv(XtX_total)

    sigma2_pooled = rss_total / dfres_total if dfres_total > 0 else np.full(V, np.nan)

    return {
        "betas": betas_pooled,
        "sigma": np.sqrt(sigma2_pooled),
        "dfres": dfres_total,
        "XtXinv": XtXinv_total,
    }
