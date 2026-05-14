"""IRLS (Iteratively Reweighted Least Squares) algorithm.

Implements robust regression by iterating between computing
weights from residuals and re-fitting the weighted least squares
problem.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ..dataset.data_access import get_run_data
from ..glm.solver import fast_lm_matrix, fast_preproject
from ..model.config import FmriLmConfig
from .estimators import bisquare_weights, huber_weights, mad_scale


def robust_refit(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: Dict,
) -> Tuple[Dict, NDArray[np.float64]]:
    """Re-fit the GLM using IRLS with robust weights.

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with robust options.
    initial_fit : dict
        Initial fit result (OLS or GLS).

    Returns
    -------
    fit_result : dict
        Updated fit result.
    weights : NDArray
        Final IRLS weights, shape ``(n_total, V)``.
    """
    robust_opts = config.robust
    max_iter = robust_opts.max_iter
    n_runs = model.n_runs  # type: ignore[attr-defined]

    # Weight function
    if robust_opts.type == "huber":
        weight_fn = lambda r, s: huber_weights(r, s, k=robust_opts.k_huber)
    elif robust_opts.type == "bisquare":
        weight_fn = lambda r, s: bisquare_weights(r, s, c=robust_opts.c_tukey)
    else:
        # Should not happen if config validation passed
        raise ValueError(f"Unknown robust type: {robust_opts.type}")

    # Get initial residuals and design matrices
    residuals_list = initial_fit["residuals"]
    run_X = initial_fit["run_X"]

    all_weights = []

    for iteration in range(max_iter):
        run_results = []
        run_projections = []
        new_residuals = []
        new_run_X = []
        iter_weights = []

        for r in range(n_runs):
            resid_r = residuals_list[r]
            X_r = run_X[r]
            Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]

            # Compute scale
            if robust_opts.scale_scope == "run":
                # Pool across voxels within run
                scale = mad_scale(resid_r, axis=0)
            elif robust_opts.scale_scope == "global":
                # Use global scale (simplified: use run scale)
                scale = mad_scale(resid_r, axis=0)
            elif robust_opts.scale_scope == "voxel":
                scale = mad_scale(resid_r, axis=0)
            else:
                scale = mad_scale(resid_r, axis=0)

            # Compute weights
            w = weight_fn(resid_r, scale)
            iter_weights.append(w)

            # Apply weights and re-fit
            w_sqrt = np.sqrt(np.maximum(w, 0.0))
            X_w = X_r * w_sqrt  # broadcast: (n,1)*(n,p) won't work
            # w_sqrt shape is (n, V) but X is (n, p) — we need row weights
            # For robust fitting, use row-wise mean weight
            w_row = np.mean(w_sqrt, axis=1)
            X_w = X_r * w_row[:, np.newaxis]
            Y_w = Y_r * w_sqrt  # voxelwise weights on Y

            # But X needs the same treatment per-voxel — this is WLS
            # Standard approach: use row weights (mean across voxels)
            Y_w = Y_r * w_row[:, np.newaxis]

            proj = fast_preproject(X_w)
            result = fast_lm_matrix(X_w, Y_w, proj, return_fitted=True)

            run_results.append(result)
            run_projections.append(proj)
            new_run_X.append(X_w)

            if result.fitted is not None:
                new_residuals.append(Y_w - result.fitted)
            else:
                new_residuals.append(Y_w - X_w @ result.betas)

        residuals_list = new_residuals
        run_X = new_run_X
        all_weights = iter_weights

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

    # Concatenate weights across runs
    weights_concat = np.vstack(all_weights) if all_weights else np.array([])

    return fit_result, weights_concat
