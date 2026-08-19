"""AR parameter estimation from residuals.

Provides Yule-Walker estimation of AR coefficients, either globally
(pooled across voxels) or per-voxel.
"""

from __future__ import annotations

from typing import Any, Optional, cast
import warnings

import numpy as np
from numpy.typing import NDArray

from .plan import WhiteningPlan


def estimate_ar_yule_walker(
    residuals: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Estimate AR parameters via Yule-Walker equations.

    Uses the autocorrelation method, pooling across columns (voxels)
    to get a robust global estimate.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix, shape ``(n, V)``.
    order : int
        AR order (e.g. 1, 2).

    Returns
    -------
    NDArray
        AR coefficients, shape ``(order,)``.
    """
    if order <= 0:
        return np.array([], dtype=np.float64)

    n, V = residuals.shape

    if n <= order:
        return np.zeros(order, dtype=np.float64)

    # Pool autocorrelations across voxels
    # r(k) = mean_v[ sum_t e(t)*e(t-k) ] / mean_v[ sum_t e(t)^2 ]
    autocorr = np.zeros(order + 1)
    for k in range(order + 1):
        if k == 0:
            autocorr[k] = np.mean(np.sum(residuals**2, axis=0))
        else:
            autocorr[k] = np.mean(np.sum(residuals[k:] * residuals[:-k], axis=0))

    # Normalise to correlation
    if autocorr[0] < 1e-15:
        return np.zeros(order, dtype=np.float64)
    rho = autocorr / autocorr[0]

    # Solve Yule-Walker: R @ phi = r
    # where R is the Toeplitz autocorrelation matrix
    R = np.zeros((order, order))
    r = np.zeros(order)
    for i in range(order):
        r[i] = rho[i + 1]
        for j in range(order):
            R[i, j] = rho[abs(i - j)]

    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        phi = np.zeros(order, dtype=np.float64)

    # Clamp to ensure stationarity
    phi = _enforce_stationarity(phi)

    return phi


def estimate_ar_voxelwise(
    residuals: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Estimate AR parameters independently for each voxel.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix, shape ``(n, V)``.
    order : int
        AR order.

    Returns
    -------
    NDArray
        AR coefficients, shape ``(order, V)``.
    """
    n, V = residuals.shape
    phi_all = np.zeros((order, V), dtype=np.float64)

    if n <= order:
        return phi_all

    for v in range(V):
        r_v = residuals[:, v]
        autocorr = np.array(
            [
                np.sum(r_v[k:] * r_v[: n - k]) if k > 0 else np.sum(r_v**2)
                for k in range(order + 1)
            ]
        )
        if autocorr[0] < 1e-15:
            continue
        rho = autocorr / autocorr[0]

        R = np.zeros((order, order))
        r = np.zeros(order)
        for i in range(order):
            r[i] = rho[i + 1]
            for j in range(order):
                R[i, j] = rho[abs(i - j)]

        try:
            phi_v = np.linalg.solve(R, r)
            phi_v = _enforce_stationarity(phi_v)
            phi_all[:, v] = phi_v
        except np.linalg.LinAlgError:
            pass

    return phi_all


def estimate_ar(
    residuals: NDArray[np.float64],
    order: int,
    voxelwise: bool = False,
    censor: Optional[NDArray[np.bool_]] = None,
    noise_pools: Optional[int] = None,
) -> NDArray[np.float64]:
    """Estimate AR parameters from residuals.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix, shape ``(n, V)``.
    order : int
        AR order.
    voxelwise : bool
        If ``True``, estimate per voxel; otherwise pool globally.
    censor : NDArray[bool], optional
        Boolean vector marking censored timepoints to exclude.
    noise_pools : int, optional
        When set and ``voxelwise=True``, quantise the per-voxel AR
        estimates into ``noise_pools`` equal-frequency bins by the
        first AR coefficient and replace each voxel's estimate with
        its bin's median. Matches Nilearn's
        ``noise_model="ar1"`` scheme (Nilearn defaults to 10 pools).
        Ignored when ``voxelwise=False``.

    Returns
    -------
    NDArray
        AR coefficients.  Shape ``(order,)`` for global or
        ``(order, V)`` for voxelwise.
    """
    if censor is not None and np.any(censor):
        residuals = residuals[~censor]

    if voxelwise:
        phi = estimate_ar_voxelwise(residuals, order)
        if noise_pools is not None and int(noise_pools) > 1:
            phi = _quantise_to_noise_pools(phi, int(noise_pools))
        return phi
    else:
        return estimate_ar_yule_walker(residuals, order)


def _quantise_to_noise_pools(
    phi: NDArray[np.float64], n_pools: int
) -> NDArray[np.float64]:
    """Quantise per-voxel AR estimates into equal-frequency noise pools.

    Mirrors Nilearn's ``run_glm`` noise-pool scheme: rank-sort voxels
    by their first AR coefficient, bin into ``n_pools`` equal-size
    groups, and replace each voxel's AR estimate with the median
    estimate from its bin. Voxels in the same bin share the same
    prewhitening operator — the variance-pooling step that
    distinguishes Nilearn's algorithm from per-voxel AR fits.
    """
    if phi.ndim != 2:
        raise ValueError(
            f"_quantise_to_noise_pools expects (order, V) input; got "
            f"shape {phi.shape}"
        )
    order, V = phi.shape
    if V == 0 or n_pools <= 1:
        return phi.copy()
    primary = phi[0]
    ranks = np.argsort(primary, kind="stable")
    pool_assignment = np.empty(V, dtype=np.int64)
    chunks = np.array_split(np.arange(V), int(n_pools))
    for pool_idx, idxs in enumerate(chunks):
        if idxs.size == 0:
            continue
        pool_assignment[ranks[idxs]] = pool_idx
    out = np.empty_like(phi)
    for pool_idx in range(int(n_pools)):
        members = np.where(pool_assignment == pool_idx)[0]
        if members.size == 0:
            continue
        representative = np.median(phi[:, members], axis=1)
        out[:, members] = representative[:, np.newaxis]
    return out


def _enforce_stationarity(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Enforce stationarity by shrinking AR coefficients if needed.

    Checks that all roots of the AR polynomial lie outside the unit
    circle.  If not, shrinks coefficients towards zero.
    """
    if len(phi) == 0:
        return phi

    # Quick check for AR(1): |phi| < 1
    if len(phi) == 1:
        return np.clip(phi, -0.99, 0.99)

    # General case: check roots
    poly_coeffs = np.concatenate([[1.0], -phi])
    roots = np.roots(poly_coeffs)
    max_root_mag = np.max(np.abs(roots)) if len(roots) > 0 else 0.0

    if max_root_mag < 1.0:
        return phi

    # Shrink towards zero until stationary
    for shrink in [0.95, 0.9, 0.8, 0.5, 0.1]:
        phi_shrunk = phi * shrink
        poly_coeffs = np.concatenate([[1.0], -phi_shrunk])
        roots = np.roots(poly_coeffs)
        if np.all(np.abs(roots) < 1.0):
            return phi_shrunk

    return np.zeros_like(phi)


# ---------------------------------------------------------------------------
# BIC-based AR order selection
# ---------------------------------------------------------------------------


def estimate_ar_bic(
    y: NDArray[np.float64],
    p_max: int,
) -> dict[str, Any]:
    """Select AR order via BIC and estimate coefficients.

    Parameters
    ----------
    y : NDArray
        1-D time series (centered).
    p_max : int
        Maximum AR order to consider.

    Returns
    -------
    dict
        ``{"phi": NDArray, "order": (p, 0)}``
    """
    from .numhelpers import enforce_stationary_ar, levinson_durbin

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y - y.mean()
    n = len(y)
    if n < 3:
        return {"phi": np.array([], dtype=np.float64), "order": (0, 0)}

    # Compute autocovariance up to p_max
    gamma = np.zeros(p_max + 1)
    for lag in range(p_max + 1):
        if lag >= n:
            break
        gamma[lag] = np.sum(y[: n - lag] * y[lag:]) / n

    if gamma[0] < 1e-15:
        return {"phi": np.array([], dtype=np.float64), "order": (0, 0)}

    log_n = np.log(n)
    # BIC for order 0 (white noise)
    sigma2_0 = max(gamma[0], 1e-15)
    best_bic = n * np.log(sigma2_0) + log_n  # 1 parameter (variance)
    best_phi = np.array([], dtype=np.float64)
    best_p = 0

    p_sel = min(int(p_max), n - 1, int(np.floor(n / 5.0)))
    for pp in range(1, max(p_sel, 0) + 1):
        if len(gamma) < pp + 1:
            break
        phi_try, sigma2 = levinson_durbin(gamma[: pp + 1], pp)
        sigma2 = max(sigma2, 1e-15)
        bic = n * np.log(sigma2) + (pp + 1) * log_n
        if bic < best_bic:
            best_bic = bic
            best_phi = enforce_stationary_ar(phi_try)
            best_p = pp

    return {"phi": best_phi, "order": (best_p, 0)}


# ---------------------------------------------------------------------------
# fit_noise: main entry point for noise estimation
# ---------------------------------------------------------------------------


def fit_noise(
    resid: Optional[NDArray[np.float64]] = None,
    Y: Optional[NDArray[np.float64]] = None,
    X: Optional[NDArray[np.float64]] = None,
    runs: Optional[NDArray[Any]] = None,
    censor: Optional[NDArray[Any]] = None,
    method: str = "ar",
    p: object = "auto",
    q: int = 0,
    p_max: int = 6,
    exact_first: str = "ar1",
    pooling: str = "global",
    parcels: Optional[NDArray[Any]] = None,
    parcel_sets: Optional[dict[str, Any]] = None,
    multiscale: object = None,
    ms_mode: Optional[str] = None,
    p_target: Optional[int] = None,
    beta: float = 0.5,
    hr_iter: int = 0,
    step1: str = "yw",
    design: Optional[NDArray[np.float64]] = None,
    acvf_correction: Optional[Any] = None,
    correction_max_lag: int = 25,
) -> "WhiteningPlan":
    """Fit an AR/ARMA noise model and return a whitening plan.

    Ports R's ``fmriAR::fit_noise()``.

    Parameters
    ----------
    resid : NDArray, optional
        Residual matrix, shape ``(n, V)``.
    Y, X : NDArray, optional
        Data and design matrices (used to compute residuals if *resid*
        is not provided).
    runs : NDArray, optional
        Integer run labels, length *n*.
    censor : NDArray, optional
        0-based indices of censored timepoints (or boolean mask).
    method : str
        ``"ar"`` or ``"arma"``.
    p : int or ``"auto"``
        AR order (``"auto"`` triggers BIC selection for AR).
    q : int
        MA order (only used when ``method="arma"``).
    p_max : int
        Maximum AR order for BIC selection.
    exact_first : str
        ``"ar1"`` to apply exact first-sample scaling, ``"none"`` otherwise.
    pooling : str
        ``"global"``, ``"run"``, or ``"parcel"``.
    parcels : NDArray, optional
        Voxel-to-parcel mapping (length *V*) for ``pooling="parcel"``.
    parcel_sets : dict, optional
        Nested parcel labels ``{"coarse", "medium", "fine"}`` for
        multi-scale pooling.
    multiscale : str or bool or None
        Multi-scale mode: ``"pacf_weighted"``, ``"acvf_pooled"``,
        ``True``, ``False``, or ``None``.
    ms_mode : str, optional
        Explicit multi-scale mode override.
    p_target : int, optional
        Target AR order for multi-scale pooling.
    beta : float
        Size exponent for multi-scale weights.
    hr_iter : int
        Hannan-Rissanen refinement iterations.
    step1 : str
        Preliminary fit method for HR: ``"burg"`` or ``"yw"``.
    design : ndarray, optional
        Design whose projection produced ``resid``. Opt-in residual ACVF
        bias correction (fmriAR 0.3.3); global/run AR only.
    acvf_correction : ndarray or list, optional
        Precomputed bias matrices from :func:`acvf_bias_matrix`. Mutually
        exclusive with ``design``.
    correction_max_lag : int
        Lag budget for residual-bias correction (default 25).

    Returns
    -------
    WhiteningPlan
    """
    from .acvf import (
        acvf_bias_matrix,
        acvf_from_pooled,
        acvf_max_lag,
        drop_unusable_corrections,
        estimate_ar_series,
        parcel_codes,
        pooled_acvf_segments,
        run_sets,
        sigma2_from_gamma_phi,
        valid_segments,
        yw_from_acvf,
    )
    from .numhelpers import enforce_stationary_ar

    # Compute residuals if not supplied
    if resid is None:
        if Y is not None and X is not None:
            Y = np.asarray(Y, dtype=np.float64)
            X = np.asarray(X, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y[:, np.newaxis]
            if X.ndim == 1:
                X = X[:, np.newaxis]
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ coef
        else:
            raise ValueError("fit_noise: supply 'resid' or both 'Y' and 'X'")

    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    if np.any(~np.isfinite(resid)):
        raise ValueError("'resid' contains NA, NaN, or Inf")

    n = resid.shape[0]
    if n < 10:
        raise ValueError("Series too short (n < 10)")

    # Validate parameters
    if method not in ("ar", "arma"):
        raise ValueError(f"method must be 'ar' or 'arma', got {method!r}")
    if pooling not in ("global", "run", "parcel"):
        raise ValueError(
            f"pooling must be 'global', 'run', or 'parcel', got {pooling!r}"
        )
    if exact_first not in ("ar1", "none"):
        exact_first = "ar1" if exact_first else "none"

    # Resolve multiscale mode
    ms_modes = ("pacf_weighted", "acvf_pooled")
    multiscale_mode = None
    if isinstance(multiscale, bool):
        if multiscale:
            multiscale_mode = ms_mode if ms_mode in ms_modes else "pacf_weighted"
    elif isinstance(multiscale, str) and multiscale in ms_modes:
        multiscale_mode = multiscale
    if ms_mode is not None and ms_mode in ms_modes:
        multiscale_mode = ms_mode

    # Normalize censor: convert boolean mask to 0-based indices
    if censor is not None:
        censor = np.asarray(censor)
        if censor.dtype == bool:
            censor = np.where(censor)[0]
        else:
            censor = np.asarray(censor, dtype=np.intp)
        censor = np.unique(censor[(censor >= 0) & (censor < n)])
        if len(censor) == 0:
            censor = None

    # Split into runs (contiguous labels only)
    named_runs = run_sets(runs, n)
    run_sets_idx = [idx for _, idx in named_runs]

    # Split censor indices by run (relative to run start)
    censor_by_run = [np.array([], dtype=np.intp) for _ in run_sets_idx]
    if censor is not None:
        for ri, idx in enumerate(run_sets_idx):
            c_in = np.intersect1d(censor, idx)
            if len(c_in):
                start = int(idx[0])
                censor_by_run[ri] = (c_in - start).astype(np.intp)

    def _rows_from_idx(mat: NDArray[Any], idx: NDArray[Any]) -> NDArray[Any]:
        """Return run rows, preferring slice views for contiguous indices."""
        if len(idx) == 0:
            return cast("NDArray[Any]", mat[idx])
        if len(idx) == 1:
            i0 = int(idx[0])
            return mat[i0 : i0 + 1]
        i0 = int(idx[0])
        i1 = int(idx[-1])
        if (i1 - i0 + 1) == len(idx) and np.all(np.diff(idx) == 1):
            return mat[i0 : i1 + 1]
        return cast("NDArray[Any]", mat[idx])

    run_mats = [_rows_from_idx(resid, idx) for idx in run_sets_idx]

    if design is not None and acvf_correction is not None:
        raise ValueError("supply either 'design' or 'acvf_correction', not both")
    corr_by_run: list[Optional[NDArray[Any]]] | None = None
    if design is not None or acvf_correction is not None:
        if pooling == "parcel":
            raise ValueError(
                "residual-bias correction is not yet supported for pooling='parcel'"
            )
        if method != "ar":
            raise ValueError("residual-bias correction applies to method='ar' only")
        if design is not None:
            mats = acvf_bias_matrix(
                design,
                runs=runs,
                censor=censor,
                max_lag=int(min(correction_max_lag, n)),
            )
            corr_by_run = drop_unusable_corrections(mats)
        else:
            if isinstance(acvf_correction, np.ndarray):
                corr_by_run = [np.asarray(acvf_correction, dtype=np.float64)] * len(
                    run_sets_idx
                )
            else:
                corr_list = list(acvf_correction)
                if len(corr_list) == 1:
                    corr_by_run = corr_list * len(run_sets_idx)
                elif len(corr_list) != len(run_sets_idx):
                    raise ValueError(
                        f"'acvf_correction' has {len(corr_list)} matrices but "
                        f"there are {len(run_sets_idx)} runs"
                    )
                else:
                    corr_by_run = [
                        None if c is None else np.asarray(c, dtype=np.float64)
                        for c in corr_list
                    ]
            corr_by_run = drop_unusable_corrections(corr_by_run)

    # --- Parcel pooling ---
    if pooling == "parcel":
        if method != "ar":
            raise ValueError("Parcel pooling currently supports method='ar' only")
        if parcels is None:
            raise ValueError("parcels must be provided for pooling='parcel'")
        parcels = parcel_codes(parcels)
        if parcels.size != resid.shape[1]:
            raise ValueError("parcels must have one entry per voxel")

        from .multiscale import (
            ms_combine_to_fine,
            ms_dispersion,
            ms_estimate_scale,
            ms_parent_maps,
            parcel_means,
        )

        seg = valid_segments(n, runs=runs, censor=censor)
        if seg.idx.size < 2:
            raise ValueError("no valid timepoints remain after censoring")

        def _estimator(y_col: NDArray[np.float64]) -> dict[str, Any]:
            return estimate_ar_series(
                y_col,
                p_max,
                p=p,
                starts0=seg.starts0,
                center_id=seg.run_id,
            )

        if p_target is None:
            if p == "auto":
                target = int(p_max)
            elif multiscale_mode is not None:
                target = min(int(p), int(p_max))
            else:
                target = int(p_max)
        else:
            target = min(int(p_target), int(p_max))

        M_fine_full = parcel_means(resid, parcels)
        M_fine = {
            k: np.asarray(v, dtype=np.float64)[seg.idx] for k, v in M_fine_full.items()
        }
        est_f = ms_estimate_scale(
            M_fine,
            _estimator,
            run_starts=seg.starts0,
            lag_max=target,
            center_id=seg.run_id,
        )

        def _pad_vec(x: NDArray[Any], n: int) -> NDArray[np.float64]:
            x = np.asarray(x, dtype=np.float64).ravel()
            out = np.zeros(n, dtype=np.float64)
            take = min(x.size, n)
            if take:
                out[:take] = x[:take]
            return out

        if parcel_sets is None:
            if multiscale_mode is None or target == 0:
                phi_parcel = {k: v for k, v in est_f["phi"].items()}
            elif multiscale_mode == "acvf_pooled":
                shrink = 0.6
                acvf_list = {
                    k: _pad_vec(g, target + 1) for k, g in est_f["acvf"].items()
                }
                if acvf_list:
                    avg_g = np.mean(np.column_stack(list(acvf_list.values())), axis=1)
                else:
                    avg_g = np.zeros(target + 1)
                phi_parcel = {}
                for k, g_pad in acvf_list.items():
                    g_mix = (1 - shrink) * g_pad + shrink * avg_g
                    phi_try, _ = yw_from_acvf(g_mix, target)
                    phi_parcel[k] = enforce_stationary_ar(phi_try)
            else:
                from .numhelpers import ar_to_pacf, pacf_to_ar

                shrink = 0.6
                kap_list = {}
                for k, phi_v in est_f["phi"].items():
                    kap = ar_to_pacf(phi_v)
                    kap_list[k] = _pad_vec(kap, target)

                if kap_list:
                    kap_mat = np.column_stack(list(kap_list.values()))
                    avg_kap = np.clip(kap_mat.mean(axis=1), -0.99, 0.99)
                else:
                    avg_kap = np.zeros(target)

                phi_parcel = {}
                for k, kap_f in kap_list.items():
                    kap_mix = (1 - shrink) * kap_f + shrink * avg_kap
                    kap_mix = np.clip(kap_mix, -0.99, 0.99)
                    phi_parcel[k] = pacf_to_ar(kap_mix)
        else:
            required = ("coarse", "medium", "fine")
            for key in required:
                if key not in parcel_sets:
                    raise ValueError(f"parcel_sets must contain '{key}'")

            parcels_coarse = parcel_codes(parcel_sets["coarse"], "parcel_sets$coarse")
            parcels_medium = parcel_codes(parcel_sets["medium"], "parcel_sets$medium")
            parcels_fine = parcel_codes(parcel_sets["fine"], "parcel_sets$fine")
            if not np.array_equal(parcels_fine, parcels):
                raise ValueError("parcel_sets['fine'] must match parcels")

            M_coarse = {
                k: np.asarray(v, dtype=np.float64)[seg.idx]
                for k, v in parcel_means(resid, parcels_coarse).items()
            }
            M_medium = {
                k: np.asarray(v, dtype=np.float64)[seg.idx]
                for k, v in parcel_means(resid, parcels_medium).items()
            }

            est_c = ms_estimate_scale(
                M_coarse,
                _estimator,
                run_starts=seg.starts0,
                lag_max=target,
                center_id=seg.run_id,
            )
            est_m = ms_estimate_scale(
                M_medium,
                _estimator,
                run_starts=seg.starts0,
                lag_max=target,
                center_id=seg.run_id,
            )

            parents = ms_parent_maps(parcels_fine, parcels_medium, parcels_coarse)

            n_runs_count = 1 if runs is None else len(named_runs)
            sizes = {
                "n_t": n,
                "n_runs": n_runs_count,
                "beta": beta,
                "coarse": {
                    str(k): int(v)
                    for k, v in zip(
                        *np.unique(parcels_coarse, return_counts=True), strict=True
                    )
                },
                "medium": {
                    str(k): int(v)
                    for k, v in zip(
                        *np.unique(parcels_medium, return_counts=True), strict=True
                    )
                },
                "fine": {
                    str(k): int(v)
                    for k, v in zip(
                        *np.unique(parcels_fine, return_counts=True), strict=True
                    )
                },
            }
            disp_list = {
                "coarse": ms_dispersion(resid, parcels_coarse),
                "medium": ms_dispersion(resid, parcels_medium),
                "fine": ms_dispersion(resid, parcels_fine),
            }

            if multiscale_mode is None:
                phi_parcel = {k: v for k, v in est_f["phi"].items()}
            else:
                phi_parcel = ms_combine_to_fine(
                    phi_by_coarse=est_c["phi"],
                    phi_by_medium=est_m["phi"],
                    phi_by_fine=est_f["phi"],
                    acvf_by_coarse=est_c.get("acvf")
                    if multiscale_mode == "acvf_pooled"
                    else None,
                    acvf_by_medium=est_m.get("acvf")
                    if multiscale_mode == "acvf_pooled"
                    else None,
                    acvf_by_fine=est_f.get("acvf")
                    if multiscale_mode == "acvf_pooled"
                    else None,
                    parents=parents,
                    sizes=sizes,
                    disp_list=disp_list,
                    p_target=target,
                    mode=multiscale_mode,
                )

        if multiscale_mode is None and p_target is not None and target > 0:
            padded: dict[str, NDArray[Any]] = {}
            for k, ph in phi_parcel.items():
                g = _pad_vec(est_f["acvf"][k], target + 1)
                phi_try, _ = yw_from_acvf(g, target)
                padded[k] = enforce_stationary_ar(phi_try) if phi_try.size else ph
            phi_parcel = padded

        # Trim shared trailing zeros so order reports the fitted length.
        eff = [
            int(np.max(np.where(np.asarray(ph) != 0)[0]) + 1)
            if np.any(np.asarray(ph) != 0)
            else 0
            for ph in phi_parcel.values()
        ]
        keep = max(eff) if eff else 0
        phi_parcel = {
            k: (
                np.asarray(ph, dtype=np.float64)[:keep]
                if keep
                else np.array([], dtype=np.float64)
            )
            for k, ph in phi_parcel.items()
        }

        theta_parcel = {k: np.array([], dtype=np.float64) for k in phi_parcel}
        resid_valid = resid[seg.idx]
        n_valid = seg.idx.size
        is_start = np.zeros(n_valid, dtype=bool)
        is_start[seg.starts0[seg.starts0 < n_valid]] = True
        is_start[0] = True
        seg_id_p = np.cumsum(is_start.astype(np.intp))
        gamma_parcel: dict[str, NDArray[Any]] = {}
        sigma2_parcel: dict[str, float] = {}
        for k, ph in phi_parcel.items():
            cols = np.where(parcels == int(k))[0]
            if cols.size == 0:
                gamma_parcel[k] = np.array([], dtype=np.float64)
                sigma2_parcel[k] = float("nan")
                continue
            lag_k = max(int(target), len(ph), 1)
            pooled_p = pooled_acvf_segments(
                resid_valid[:, cols], seg_id_p, lag_k, center_id=seg.run_id
            )
            g, _ = acvf_from_pooled(pooled_p, order=lag_k)
            gamma_parcel[k] = g
            sigma2_parcel[k] = sigma2_from_gamma_phi(g, ph)

        return WhiteningPlan(
            phi=None,
            theta=None,
            order=(keep, 0),
            runs=runs,
            exact_first=(exact_first == "ar1"),
            method=method,
            pooling="parcel",
            parcels=parcels,
            parcel_ids=list(phi_parcel.keys()),
            phi_by_parcel=phi_parcel,
            theta_by_parcel=theta_parcel,
            censor=censor,
            gamma_by_parcel=gamma_parcel,
            sigma2_by_parcel=sigma2_parcel,
        )

    # --- Per-run estimation ---
    def _est_run(
        mat: NDArray[Any],
        censor_rel: NDArray[Any],
        corr: Optional[NDArray[Any]] = None,
    ) -> dict[str, Any]:
        n_run = mat.shape[0]
        if len(censor_rel) > 0:
            valid = np.ones(n_run, dtype=bool)
            valid[censor_rel] = False
            valid_idx = np.where(valid)[0]
        else:
            valid_idx = np.arange(n_run)

        empty: dict[str, Any] = {
            "phi": np.array([], dtype=np.float64),
            "theta": np.array([], dtype=np.float64),
            "order": (0, 0),
            "gamma": np.array([], dtype=np.float64),
            "sigma2": float("nan"),
        }

        if method == "arma":
            if (
                len(censor_rel) > 0
                and valid_idx.size > 1
                and np.any(np.diff(valid_idx) > 1)
            ):
                warnings.warn(
                    "fit_noise: method='arma' estimates across censoring gaps; "
                    "AR and MA coefficients will be biased. Prefer method='ar' "
                    "when censoring is present.",
                    UserWarning,
                    stacklevel=2,
                )
            mat_valid = mat[valid_idx]
            y_mean = mat_valid.mean(axis=1)
            from .hr_arma import hr_arma

            pp = min(2, p_max) if p == "auto" else int(cast(Any, p))
            qq = int(q)
            fit = hr_arma(y_mean, p=pp, q=qq, n_iter=hr_iter, step1=step1)
            seg_id_ma = np.cumsum(
                np.concatenate([[1], (np.diff(valid_idx) > 1).astype(np.intp)])
            )
            lag_ma = max(1, pp + qq)
            pooled_ma = pooled_acvf_segments(mat_valid, seg_id_ma, lag_ma)
            g_ma, _ = acvf_from_pooled(pooled_ma, order=lag_ma)
            fit["gamma"] = g_ma
            fit["sigma2"] = float("nan")
            return fit

        if method != "ar":
            raise ValueError(f"Unknown estimation method: {method!r}")

        n_eff = int(len(valid_idx))
        if n_eff <= 1:
            return empty
        p_cap = min(int(p_max), n_eff - 1)
        if p_cap < 1:
            return empty

        seg_id = np.cumsum(
            np.concatenate([[1], (np.diff(valid_idx) > 1).astype(np.intp)])
        )
        if corr is None:
            lag_budget = p_cap
        else:
            lag_budget = min(max(p_cap, int(np.asarray(corr).shape[0]) - 1), n_eff - 1)
        pooled = pooled_acvf_segments(mat[valid_idx], seg_id, lag_budget)
        gamma0, _ = acvf_from_pooled(pooled, order=0, correction=corr)
        if gamma0.size == 0 or not np.isfinite(gamma0[0]) or gamma0[0] <= 0:
            return empty
        p_cap = min(p_cap, acvf_max_lag(pooled))
        if p_cap < 1:
            empty = dict(empty)
            empty["gamma"] = gamma0
            empty["sigma2"] = float(gamma0[0])
            return empty
        gamma_full, _ = acvf_from_pooled(pooled, order=p_cap, correction=corr)

        def _fit_order(pp: int) -> tuple[NDArray[np.float64], float]:
            g, _ = acvf_from_pooled(pooled, order=pp, correction=corr)
            phi, sigma2 = yw_from_acvf(g[: pp + 1], pp)
            return enforce_stationary_ar(phi, 0.99), float(max(sigma2, 1e-12))

        if p != "auto":
            pp = min(int(cast(Any, p)), p_cap)
            if pp <= 0:
                empty = dict(empty)
                empty["gamma"] = gamma_full
                empty["sigma2"] = (
                    float(gamma_full[0]) if gamma_full.size else float("nan")
                )
                return empty
            phi, sigma2 = _fit_order(pp)
            return {
                "phi": phi,
                "theta": np.array([], dtype=np.float64),
                "order": (pp, 0),
                "gamma": gamma_full,
                "sigma2": sigma2,
            }

        n_eff_log = np.log(n_eff)
        sigma0 = max(float(gamma0[0]), 1e-12)
        best_bic = 2.0 * n_eff * np.log(sigma0) + n_eff_log
        best_phi = np.array([], dtype=np.float64)
        best_p = 0
        best_sigma2 = sigma0
        p_sel = min(p_cap, int(np.floor(n_eff / 5.0)))
        for pp in range(1, max(p_sel, 0) + 1):
            phi_pp, sigma2 = _fit_order(pp)
            if not np.isfinite(sigma2):
                continue
            bic = 2.0 * n_eff * np.log(sigma2) + (pp + 1) * n_eff_log
            if not np.isfinite(bic) or bic >= best_bic:
                continue
            if phi_pp.size != pp or not np.all(np.isfinite(phi_pp)):
                continue
            best_bic = bic
            best_phi = phi_pp
            best_p = pp
            best_sigma2 = sigma2
        return {
            "phi": best_phi,
            "theta": np.array([], dtype=np.float64),
            "order": (best_p, 0),
            "gamma": gamma_full,
            "sigma2": best_sigma2,
        }

    if corr_by_run is None:
        estimates = [
            _est_run(m, c) for m, c in zip(run_mats, censor_by_run, strict=True)
        ]
    else:
        estimates = [
            _est_run(m, c, corr)
            for m, c, corr in zip(run_mats, censor_by_run, corr_by_run, strict=True)
        ]

    from .numhelpers import enforce_invertible_ma

    # --- Pool across runs ---
    if pooling == "global":
        lens = np.array(
            [
                int(len(rs) - len(c))
                for rs, c in zip(run_sets_idx, censor_by_run, strict=True)
            ],
            dtype=np.float64,
        )
        if float(lens.sum()) <= 0:
            raise ValueError("no valid timepoints remain after censoring")
        w = lens / lens.sum()
        pmax_len = max(len(e["phi"]) for e in estimates) if estimates else 0
        qmax_len = max(len(e.get("theta", [])) for e in estimates) if estimates else 0

        Phi = np.zeros((len(estimates), max(pmax_len, 1)))
        Th = np.zeros((len(estimates), max(qmax_len, 1)))
        for i, e in enumerate(estimates):
            phi_e = e["phi"]
            if len(phi_e):
                Phi[i, : len(phi_e)] = phi_e
            theta_e = e.get("theta", np.array([]))
            if len(theta_e):
                Th[i, : len(theta_e)] = theta_e

        phi_avg = (
            (w @ Phi)[:pmax_len] if pmax_len > 0 else np.array([], dtype=np.float64)
        )
        theta_avg = (
            (w @ Th)[:qmax_len] if qmax_len > 0 else np.array([], dtype=np.float64)
        )
        if phi_avg.size:
            phi_avg = enforce_stationary_ar(phi_avg, 0.99)
        if theta_avg.size:
            theta_avg = enforce_invertible_ma(theta_avg)
        phi_list = [phi_avg]
        theta_list = [theta_avg]

        glens = np.array([len(e.get("gamma", [])) for e in estimates], dtype=np.intp)
        has_g = glens > 0
        if np.any(has_g):
            gmin = int(np.min(glens[has_g]))
            G = np.vstack(
                [
                    e["gamma"][:gmin]
                    for e, ok in zip(estimates, has_g, strict=True)
                    if ok
                ]
            )
            wg = lens[has_g] / lens[has_g].sum()
            gamma_list = [np.asarray(wg @ G, dtype=np.float64)]
        else:
            gamma_list = [np.array([], dtype=np.float64)]
        if method == "arma":
            sigma2_list = [float("nan")]
        else:
            sigma2_list = [sigma2_from_gamma_phi(gamma_list[0], phi_avg)]
    else:
        phi_list = [e["phi"] for e in estimates]
        theta_list = [e.get("theta", np.array([], dtype=np.float64)) for e in estimates]
        gamma_list = [
            np.asarray(e.get("gamma", np.array([], dtype=np.float64)), dtype=np.float64)
            for e in estimates
        ]
        if method == "arma":
            sigma2_list = [float("nan") for _ in estimates]
        else:
            sigma2_list = [
                sigma2_from_gamma_phi(g, ph)
                for g, ph in zip(gamma_list, phi_list, strict=True)
            ]

    if pooling == "global":
        order_p = len(phi_list[0]) if phi_list else 0
        order_q = len(theta_list[0]) if theta_list else 0
    else:
        order_p = max(len(ph) for ph in phi_list) if phi_list else 0
        order_q = max(len(th) for th in theta_list) if theta_list else 0

    return WhiteningPlan(
        phi=phi_list,
        theta=theta_list,
        order=(order_p, order_q),
        runs=runs,
        exact_first=(exact_first == "ar1"),
        method=method,
        pooling=pooling,
        censor=censor,
        gamma=gamma_list,
        sigma2=sigma2_list,
    )
