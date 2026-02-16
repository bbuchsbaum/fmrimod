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

    # 1. Censoring
    if censor is not None and np.any(censor):
        X_fit, Y_fit, _ = apply_censoring(X_fit, Y_fit, censor)

    # 2. Volume weighting
    if config.volume_weights.enabled:
        if config.volume_weights.weights is not None:
            weights = config.volume_weights.weights
            if censor is not None and np.any(censor):
                weights = weights[~censor]
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
            lam_val = 1.0  # TODO: implement auto/gcv selection
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

    # Weighted combination: weight_r = 1/sigma2_r for each voxel
    # For betas: B_pool = (sum_r XtXinv_r^{-1} B_r) / (sum_r XtXinv_r^{-1})
    # Simpler: sum_r XtX_r and sum_r XtX_r B_r
    p_dim = results[0].betas.shape[0]
    V = results[0].betas.shape[1]

    # Accumulate X'X and X'X @ B across runs
    XtX_total = np.zeros((p_dim, p_dim))
    XtXB_total = np.zeros((p_dim, V))
    rss_total = np.zeros(V)
    dfres_total = 0.0

    for r, proj in zip(results, projections):
        # XtX_r = inv(XtXinv_r)
        try:
            XtX_r = np.linalg.inv(proj.XtXinv)
        except np.linalg.LinAlgError:
            XtX_r = np.linalg.pinv(proj.XtXinv)

        XtX_total += XtX_r
        XtXB_total += XtX_r @ r.betas
        rss_total += r.rss
        dfres_total += r.dfres

    # Pooled (X'X)^{-1}
    try:
        XtXinv_total = np.linalg.inv(XtX_total)
    except np.linalg.LinAlgError:
        XtXinv_total = np.linalg.pinv(XtX_total)

    betas_pooled = XtXinv_total @ XtXB_total
    sigma2_pooled = rss_total / dfres_total if dfres_total > 0 else np.full(V, np.nan)

    return {
        "betas": betas_pooled,
        "sigma": np.sqrt(sigma2_pooled),
        "dfres": dfres_total,
        "XtXinv": XtXinv_total,
    }
