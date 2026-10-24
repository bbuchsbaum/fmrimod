"""IRLS (Iteratively Reweighted Least Squares) algorithm.

Implements robust regression by iterating between computing
weights from residuals and re-fitting the weighted least squares
problem.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmrimod.glm.solver import LmResult, Projection, fast_lm_matrix, fast_preproject
from fmrimod.model.config import FmriLmConfig, RobustOptions

from .estimators import bisquare_weights, huber_weights

# fmrireg robust_iterative_fitter convergence defaults (relative change in
# the coefficients, checked from the first iteration on).
_TOL = 1e-4
_MIN_ITER = 1
_EPS = float(np.finfo(np.float64).eps)
# The psi functions take residuals already divided by the robust scale.
_UNIT_SCALE = np.ones(1)


def robust_refit(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: dict[str, Any],
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Re-fit the GLM using IRLS with robust weights.

    Each run is refitted on the matrices the initial fit actually solved
    (``initial_fit["run_X"]`` and the data they were paired with: the
    preprocessed run, or the whitened run when AR ran). Every IRLS step
    weights those base matrices by one row-weight vector, so the result is
    exactly the weighted least-squares fit with the returned weights
    (fmrireg ``robust_iterative_fitter``, fb00ab1).

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with robust options.
    initial_fit : dict
        Initial fit result (OLS or GLS) with per-run ``run_X``,
        ``residuals`` and ``run_results``.

    Returns
    -------
    fit_result : dict
        Updated fit result. Each run's residual df is the number of
        observations with non-zero weight minus the rank of the weighted
        design (``stats::lm.wfit``).
    weights : NDArray
        Row weights used for the final solve, shape ``(n_total,)``,
        concatenated over runs. Rows that carried no observation (zero
        prior volume weight) are reported as 0.
    """
    robust_opts = config.robust
    n_runs = model.n_runs  # type: ignore[attr-defined]
    if robust_opts.type not in ("huber", "bisquare"):
        # Should not happen if config validation passed
        raise ValueError(f"Unknown robust type: {robust_opts.type}")

    residuals_list = initial_fit["residuals"]
    run_X = initial_fit["run_X"]
    initial_results = initial_fit["run_results"]

    # scale_scope="global": one scale from the initial residuals of all runs.
    sigma_fixed: float | None = None
    if robust_opts.scale_scope == "global":
        row_medians = np.concatenate(
            [np.median(np.abs(residuals_list[r]), axis=1) for r in range(n_runs)]
        )
        sigma_fixed = _floor_scale(1.4826 * float(np.median(row_medians)))

    run_results = []
    run_projections = []
    new_residuals = []
    new_run_X = []
    all_weights = []

    for r in range(n_runs):
        X0 = run_X[r]
        initial = initial_results[r]
        # The data the initial fit paired with X0: residuals + fitted values.
        Y0 = residuals_list[r] + X0 @ initial.betas
        obs_weights = getattr(initial, "obs_weights", None)
        active = (
            np.ones(X0.shape[0], dtype=bool)
            if obs_weights is None
            else np.asarray(obs_weights) > 0
        )

        betas = initial.betas
        for iteration in range(robust_opts.max_iter):
            betas_prev = betas
            u = _scaled_row_residuals(
                Y0 - X0 @ betas, robust_opts.scale_scope, sigma_fixed
            )
            w = _psi(u, robust_opts)
            sqrt_w = np.sqrt(w)[:, np.newaxis]
            X_w = X0 * sqrt_w
            Y_w = Y0 * sqrt_w
            proj = fast_preproject(X_w, check_finite=False, method=config.solver)
            result = fast_lm_matrix(X_w, Y_w, proj, return_fitted=True, check_finite=False)
            betas = result.betas
            if iteration + 1 >= _MIN_ITER:
                scale = max(1.0, float(np.max(np.abs(betas_prev))))
                delta = float(np.max(np.abs(betas - betas_prev))) / scale
                if np.isfinite(delta) and delta <= _TOL:
                    break

        # A zero-weight row carries no information (stats::lm.wfit).
        dfres = float(np.count_nonzero((w > 0) & active) - proj.rank)
        proj, result = _with_dfres(proj, result, dfres)
        result.obs_weights = obs_weights

        run_results.append(result)
        run_projections.append(proj)
        new_run_X.append(X_w)
        assert result.fitted is not None
        new_residuals.append(Y_w - result.fitted)
        all_weights.append(np.where(active, w, 0.0))

    # Pool final results
    from ..glm.strategies import _pool_run_results

    pooled = _pool_run_results(run_results, run_projections)

    fit_result = {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": run_projections,
        "run_results": run_results,
        "residuals": new_residuals,
        "run_X": new_run_X,
    }

    return fit_result, np.concatenate(all_weights)


def _psi(u: NDArray[np.float64], opts: RobustOptions) -> NDArray[np.float64]:
    """Row weights from scaled residuals (Huber or Tukey bisquare)."""
    u2 = u[:, np.newaxis]
    if opts.type == "huber":
        return huber_weights(u2, _UNIT_SCALE, k=opts.k_huber)[:, 0]
    return bisquare_weights(u2, _UNIT_SCALE, c=opts.c_tukey)[:, 0]


def _floor_scale(sigma: float) -> float:
    """fmrireg's guard against a zero or non-finite robust scale."""
    return sigma if np.isfinite(sigma) and sigma > _EPS else _EPS


def _scaled_row_residuals(
    resid: NDArray[np.float64],
    scope: str,
    sigma_fixed: float | None,
) -> NDArray[np.float64]:
    """Per-row scaled residual ``u`` that drives the row weights.

    ``u_i = median_v(|r_iv| / sigma_v)`` with ``sigma`` the MAD-type scale
    ``1.4826 * median|r|`` taken per voxel (``"voxel"``), over the run
    (``"run"``: median of the row medians) or fixed across runs
    (``"global"``), as fmrireg ``robust_iterative_fitter``.
    """
    abs_resid = np.abs(resid)
    if sigma_fixed is not None:
        return np.median(abs_resid, axis=1) / sigma_fixed
    if scope == "voxel":
        sigma_v = 1.4826 * np.median(abs_resid, axis=0)
        sigma_v = np.where(np.isfinite(sigma_v) & (sigma_v > _EPS), sigma_v, _EPS)
        return np.median(abs_resid / sigma_v[np.newaxis, :], axis=1)
    row_medians = np.median(abs_resid, axis=1)
    return row_medians / _floor_scale(1.4826 * float(np.median(row_medians)))


def _with_dfres(
    proj: Projection,
    result: LmResult,
    dfres: float,
) -> tuple[Projection, LmResult]:
    """Re-express a WLS solve with its own residual df."""
    if dfres == proj.dfres:
        return proj, result
    sigma2 = (
        result.rss / dfres if dfres > 0 else np.full_like(result.rss, np.nan)
    )
    return replace(proj, dfres=dfres), replace(result, dfres=dfres, sigma2=sigma2)
