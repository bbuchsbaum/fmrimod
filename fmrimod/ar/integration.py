"""Iterative GLS: alternates AR estimation and GLM fitting.

Implements the standard two-stage (or multi-stage) approach:
1. Fit OLS to get residuals
2. Estimate AR parameters from residuals
3. Whiten design and data
4. Re-fit OLS on whitened data
5. (Optionally repeat)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..model.config import FmriLmConfig
from .estimation import estimate_ar
from .whitening import ar_whiten_matrix
from ..glm.solver import fast_preproject, fast_lm_matrix


def iterative_gls(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: Dict,
) -> Tuple[Dict, NDArray[np.float64]]:
    """Run iterative GLS with AR whitening.

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with AR options.
    initial_fit : dict
        Initial OLS fit result from ``fit_runwise()``.

    Returns
    -------
    fit_result : dict
        Updated fit result with whitened estimates.
    ar_params : NDArray
        Estimated AR parameters.
    """
    ar_opts = config.ar
    ar_order = ar_opts.ar_order
    n_iter = ar_opts.iter_gls
    n_runs = model.n_runs  # type: ignore[attr-defined]

    # Current residuals from initial fit
    residuals_list = initial_fit["residuals"]
    run_X = initial_fit["run_X"]

    # Storage for per-run AR params
    ar_params_per_run = []

    current_fit = initial_fit

    for iteration in range(n_iter):
        # Step 1: Estimate AR from current residuals
        run_ar_params = []
        for r in range(n_runs):
            resid_r = residuals_list[r]
            censor_r = None
            if hasattr(model, "dataset") and hasattr(model.dataset, "get_censor"):  # type: ignore[attr-defined]
                censor_r = model.dataset.get_censor(r)  # type: ignore[attr-defined]

            phi_r = estimate_ar(
                resid_r,
                order=ar_order,
                voxelwise=ar_opts.voxelwise,
                censor=censor_r,
            )
            run_ar_params.append(phi_r)

        # Step 2: Whiten and re-fit each run
        run_results = []
        run_projections = []
        new_residuals = []
        ar_residuals = []
        new_run_X = []

        for r in range(n_runs):
            Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
            X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]
            phi_r = run_ar_params[r]

            # Whiten
            X_w, Y_w = ar_whiten_matrix(
                X_r,
                Y_r,
                phi_r,
                exact_first_ar1=ar_opts.exact_first,
            )

            # Re-fit
            proj = fast_preproject(X_w)
            result = fast_lm_matrix(X_w, Y_w, proj, return_fitted=True)

            run_results.append(result)
            run_projections.append(proj)
            new_run_X.append(X_w)

            # Residuals on whitened scale (kept in fit payload)
            if result.fitted is not None:
                new_residuals.append(Y_w - result.fitted)
            else:
                new_residuals.append(Y_w - X_w @ result.betas)
            # Residuals on original scale (used for next AR update)
            ar_residuals.append(Y_r - X_r @ result.betas)

        # Update for next iteration
        residuals_list = ar_residuals

        # Pool results
        from ..glm.strategies import _pool_run_results

        pooled = _pool_run_results(run_results, run_projections)

        current_fit = {
            "betas": pooled["betas"],
            "sigma": pooled["sigma"],
            "dfres": pooled["dfres"],
            "XtXinv": pooled["XtXinv"],
            "projections": run_projections,
            "run_results": run_results,
            "residuals": new_residuals,
            "run_X": new_run_X,
        }
        ar_params_per_run = run_ar_params

    # Combine AR params honoring config.global_ar.
    if not ar_params_per_run:
        ar_params = np.array([])
    elif len(ar_params_per_run) == 1:
        ar_params = ar_params_per_run[0]
    elif ar_opts.global_ar:
        ar_params = np.mean(np.stack(ar_params_per_run, axis=0), axis=0)
    else:
        ar_params = np.stack(ar_params_per_run, axis=0)

    return current_fit, ar_params


def iterative_ar_gls(
    model: object,  # FmriModel
    config: FmriLmConfig,
    initial_fit: Dict,
) -> Tuple[Dict, object]:
    """Run iterative GLS using plan-based whitening.

    Enhanced version of :func:`iterative_gls` that uses
    :func:`~fmrimod.ar.estimation.fit_noise` and
    :func:`~fmrimod.ar.whitening.whiten_apply` for full ARMA
    support, segment-aware whitening, and convergence checking.

    Parameters
    ----------
    model : FmriModel
        The fMRI model.
    config : FmriLmConfig
        Configuration with AR options.
    initial_fit : dict
        Initial OLS fit result from ``fit_runwise()``.

    Returns
    -------
    fit_result : dict
        Updated fit result with whitened estimates.
    plan : WhiteningPlan
        Final whitening plan.
    """
    from .estimation import fit_noise
    from .whitening import whiten_apply

    ar_opts = config.ar
    n_iter = ar_opts.iter_gls
    n_runs = model.n_runs  # type: ignore[attr-defined]

    # Map config to fit_noise kwargs
    method = getattr(ar_opts, "method", "ar")
    q = getattr(ar_opts, "q", 0)
    p_max_val = getattr(ar_opts, "p_max", 6)
    pooling = getattr(ar_opts, "pooling", "global")
    parcels = getattr(ar_opts, "parcels", None)
    convergence_tol = getattr(ar_opts, "convergence_tol", 5e-3)

    # Determine p argument
    ar_order = ar_opts.ar_order
    p_arg: object = ar_order if ar_order > 0 else "auto"

    # Build run labels
    run_labels = []
    offset = 0
    for r in range(n_runs):
        Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
        n_r = Y_r.shape[0]
        run_labels.extend([r] * n_r)
        offset += n_r
    runs_arr = np.array(run_labels, dtype=np.intp)

    # Get censor info
    censor_list = []
    offset = 0
    for r in range(n_runs):
        Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
        n_r = Y_r.shape[0]
        censor_r = None
        if hasattr(model, "dataset") and hasattr(model.dataset, "get_censor"):  # type: ignore[attr-defined]
            censor_r = model.dataset.get_censor(r)  # type: ignore[attr-defined]
        if censor_r is not None and np.any(censor_r):
            if censor_r.dtype == bool:
                indices = np.where(censor_r)[0] + offset
            else:
                indices = np.asarray(censor_r, dtype=np.intp) + offset
            censor_list.append(indices)
        offset += n_r
    censor_arr = np.concatenate(censor_list) if censor_list else None

    residuals_list = initial_fit["residuals"]
    current_fit = initial_fit
    prev_plan = None

    for iteration in range(n_iter):
        # Stack residuals across runs
        resid_full = np.vstack(residuals_list)

        # Fit noise model
        plan = fit_noise(
            resid=resid_full,
            runs=runs_arr,
            censor=censor_arr,
            method=method,
            p=p_arg,
            q=q,
            p_max=p_max_val,
            pooling=pooling,
            parcels=parcels,
            exact_first="ar1" if ar_opts.exact_first else "none",
        )

        # Convergence check
        if prev_plan is not None and convergence_tol > 0:
            converged = _check_convergence(prev_plan, plan, convergence_tol)
            if converged:
                break
        prev_plan = plan

        # Whiten and re-fit each run
        run_results = []
        run_projections = []
        new_residuals = []
        ar_residuals = []
        new_run_X = []

        for r in range(n_runs):
            Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
            X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

            # Create a sub-plan for this run
            run_mask = runs_arr == r
            run_runs = np.ones(int(run_mask.sum()), dtype=np.intp)

            # Get run's phi/theta from the plan
            if plan.pooling == "parcel":
                # For parcel plans, use whiten_apply directly
                wr = whiten_apply(plan, X_r, Y_r, runs=run_runs, parcels=parcels)
                X_w = wr.X
                Y_w = wr.Y
                # For parcel plan, X_w is None; use first parcel's X
                if X_w is None and wr.X_by:
                    X_w = next(iter(wr.X_by.values()))
            else:
                phi_idx = 0 if plan.pooling == "global" else min(r, len(plan.phi) - 1)
                phi_r = plan.phi[phi_idx] if plan.phi else np.array([])
                theta_r = (plan.theta[phi_idx]
                           if plan.theta and phi_idx < len(plan.theta)
                           else np.array([]))

                from .whitening import arma_whiten_segments
                seg_starts = np.array([0], dtype=np.intp)
                X_w = arma_whiten_segments(X_r, phi_r, theta_r, seg_starts,
                                           exact_first_ar1=plan.exact_first)
                Y_w = arma_whiten_segments(Y_r, phi_r, theta_r, seg_starts,
                                           exact_first_ar1=plan.exact_first)

            # Re-fit
            proj = fast_preproject(X_w)
            result = fast_lm_matrix(X_w, Y_w, proj, return_fitted=True)

            run_results.append(result)
            run_projections.append(proj)
            new_run_X.append(X_w)

            # Residuals on whitened scale (kept in fit payload)
            if result.fitted is not None:
                new_residuals.append(Y_w - result.fitted)
            else:
                new_residuals.append(Y_w - X_w @ result.betas)
            # Residuals on original scale (used for next AR update)
            ar_residuals.append(Y_r - X_r @ result.betas)

        # Update for next iteration
        residuals_list = ar_residuals

        # Pool results
        from ..glm.strategies import _pool_run_results

        pooled = _pool_run_results(run_results, run_projections)

        current_fit = {
            "betas": pooled["betas"],
            "sigma": pooled["sigma"],
            "dfres": pooled["dfres"],
            "XtXinv": pooled["XtXinv"],
            "projections": run_projections,
            "run_results": run_results,
            "residuals": new_residuals,
            "run_X": new_run_X,
        }

    return current_fit, plan


def _check_convergence(prev_plan, new_plan, tol: float) -> bool:
    """Check if AR parameters have converged between iterations."""
    try:
        if prev_plan.phi is not None and new_plan.phi is not None:
            for phi_old, phi_new in zip(prev_plan.phi, new_plan.phi):
                if len(phi_old) == 0 and len(phi_new) == 0:
                    continue
                if len(phi_old) != len(phi_new):
                    return False
                if np.max(np.abs(phi_new - phi_old)) >= tol:
                    return False
            return True
        if prev_plan.phi_by_parcel and new_plan.phi_by_parcel:
            for key in prev_plan.phi_by_parcel:
                if key not in new_plan.phi_by_parcel:
                    return False
                old = prev_plan.phi_by_parcel[key]
                new = new_plan.phi_by_parcel[key]
                if len(old) != len(new):
                    return False
                if len(old) > 0 and np.max(np.abs(new - old)) >= tol:
                    return False
            return True
    except Exception:
        pass
    return False
