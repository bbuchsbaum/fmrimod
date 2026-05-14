"""AR parameter estimation from residuals.

Provides Yule-Walker estimation of AR coefficients, either globally
(pooled across voxels) or per-voxel.
"""

from __future__ import annotations

from typing import Optional

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
            autocorr[k] = np.mean(np.sum(residuals ** 2, axis=0))
        else:
            autocorr[k] = np.mean(
                np.sum(residuals[k:] * residuals[:-k], axis=0)
            )

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
        autocorr = np.array([
            np.sum(r_v[k:] * r_v[:n - k]) if k > 0 else np.sum(r_v ** 2)
            for k in range(order + 1)
        ])
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

    Returns
    -------
    NDArray
        AR coefficients.  Shape ``(order,)`` for global or
        ``(order, V)`` for voxelwise.
    """
    if censor is not None and np.any(censor):
        residuals = residuals[~censor]

    if voxelwise:
        return estimate_ar_voxelwise(residuals, order)
    else:
        return estimate_ar_yule_walker(residuals, order)


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
) -> dict:
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

    for pp in range(1, min(p_max, n - 1) + 1):
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
    runs: Optional[NDArray] = None,
    censor: Optional[NDArray] = None,
    method: str = "ar",
    p: object = "auto",
    q: int = 0,
    p_max: int = 6,
    exact_first: str = "ar1",
    pooling: str = "global",
    parcels: Optional[NDArray] = None,
    parcel_sets: Optional[dict] = None,
    multiscale: object = None,
    ms_mode: Optional[str] = None,
    p_target: Optional[int] = None,
    beta: float = 0.5,
    hr_iter: int = 0,
    step1: str = "yw",
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

    Returns
    -------
    WhiteningPlan
    """
    from .numhelpers import (
        enforce_stationary_ar,
        levinson_durbin,
        run_avg_acvf,
        segmented_acvf,
    )

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

    n = resid.shape[0]
    if n < 10:
        raise ValueError("Series too short (n < 10)")

    # Validate parameters
    if method not in ("ar", "arma"):
        raise ValueError(f"method must be 'ar' or 'arma', got {method!r}")
    if pooling not in ("global", "run", "parcel"):
        raise ValueError(f"pooling must be 'global', 'run', or 'parcel', got {pooling!r}")
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

    # Split into runs
    if runs is not None:
        runs = np.asarray(runs, dtype=np.intp)
        run_labels = np.unique(runs)
        run_sets = [np.where(runs == rl)[0] for rl in run_labels]
    else:
        run_sets = [np.arange(n)]

    # Split censor indices by run (relative to run start)
    censor_by_run = [np.array([], dtype=np.intp) for _ in run_sets]
    if censor is not None:
        for ri, idx in enumerate(run_sets):
            start = idx[0]
            c_in = np.intersect1d(censor, idx)
            if len(c_in):
                censor_by_run[ri] = c_in - start

    def _rows_from_idx(mat: NDArray, idx: NDArray) -> NDArray:
        """Return run rows, preferring slice views for contiguous indices."""
        if len(idx) == 0:
            return mat[idx]
        if len(idx) == 1:
            i0 = int(idx[0])
            return mat[i0 : i0 + 1]
        i0 = int(idx[0])
        i1 = int(idx[-1])
        # Fast contiguous check: run labels are typically contiguous blocks.
        if (i1 - i0 + 1) == len(idx) and np.all(np.diff(idx) == 1):
            return mat[i0 : i1 + 1]
        return mat[idx]

    run_mats = [_rows_from_idx(resid, idx) for idx in run_sets]

    # --- Parcel pooling ---
    if pooling == "parcel":
        if method != "ar":
            raise ValueError("Parcel pooling currently supports method='ar' only")
        if parcels is None:
            raise ValueError("parcels must be provided for pooling='parcel'")
        parcels = np.asarray(parcels, dtype=np.intp)

        from .multiscale import (
            ms_combine_to_fine,
            ms_dispersion,
            ms_estimate_scale,
            ms_parent_maps,
            parcel_means,
        )

        target = p_max if p_target is None else min(int(p_target), p_max)

        def _estimator(y_col):
            return estimate_ar_bic(y_col, p_max)

        M_fine = parcel_means(resid, parcels)
        est_f = ms_estimate_scale(M_fine, _estimator)

        if parcel_sets is None:
            # Single-scale parcel pooling
            if multiscale_mode is None or target == 0:
                phi_parcel = {k: v for k, v in est_f["phi"].items()}
            else:
                from .numhelpers import ar_to_pacf, pacf_to_ar
                # Shrink toward global average
                shrink = 0.6
                kap_list = {}
                for k, phi_v in est_f["phi"].items():
                    kap = ar_to_pacf(phi_v)
                    padded = np.zeros(target)
                    padded[: len(kap)] = kap
                    kap_list[k] = padded

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
            # Multi-scale pooling
            required = ("coarse", "medium", "fine")
            for key in required:
                if key not in parcel_sets:
                    raise ValueError(f"parcel_sets must contain '{key}'")

            parcels_coarse = np.asarray(parcel_sets["coarse"], dtype=np.intp)
            parcels_medium = np.asarray(parcel_sets["medium"], dtype=np.intp)
            parcels_fine = np.asarray(parcel_sets["fine"], dtype=np.intp)

            M_coarse = parcel_means(resid, parcels_coarse)
            M_medium = parcel_means(resid, parcels_medium)

            est_c = ms_estimate_scale(M_coarse, _estimator)
            est_m = ms_estimate_scale(M_medium, _estimator)

            parents = ms_parent_maps(parcels_fine, parcels_medium, parcels_coarse)

            n_runs_count = 1 if runs is None else len(np.unique(runs))
            sizes = {
                "n_t": n,
                "n_runs": n_runs_count,
                "beta": beta,
                "coarse": {str(k): int(v) for k, v in zip(*np.unique(parcels_coarse, return_counts=True))},
                "medium": {str(k): int(v) for k, v in zip(*np.unique(parcels_medium, return_counts=True))},
                "fine": {str(k): int(v) for k, v in zip(*np.unique(parcels_fine, return_counts=True))},
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
                    acvf_by_coarse=est_c.get("acvf") if multiscale_mode == "acvf_pooled" else None,
                    acvf_by_medium=est_m.get("acvf") if multiscale_mode == "acvf_pooled" else None,
                    acvf_by_fine=est_f.get("acvf") if multiscale_mode == "acvf_pooled" else None,
                    parents=parents,
                    sizes=sizes,
                    disp_list=disp_list,
                    p_target=target,
                    mode=multiscale_mode,
                )

        theta_parcel = {k: np.array([], dtype=np.float64) for k in phi_parcel}
        max_p = max((len(v) for v in phi_parcel.values()), default=0)

        return WhiteningPlan(
            phi=None,
            theta=None,
            order=(max_p, 0),
            runs=runs,
            exact_first=(exact_first == "ar1"),
            method=method,
            pooling="parcel",
            parcels=parcels,
            parcel_ids=list(phi_parcel.keys()),
            phi_by_parcel=phi_parcel,
            theta_by_parcel=theta_parcel,
            censor=censor,
        )

    # --- Per-run estimation ---
    def _est_run(mat, censor_rel):
        n_run = mat.shape[0]

        if method == "arma":
            # ARMA uses run-mean time series; avoid materializing full-row copies
            # when there is no censoring.
            if len(censor_rel) > 0:
                valid = np.ones(n_run, dtype=bool)
                valid[censor_rel] = False
                y_mean = mat[valid].mean(axis=1)
            else:
                y_mean = mat.mean(axis=1)

            from .hr_arma import hr_arma
            pp = min(2, p_max) if p == "auto" else int(p)
            qq = int(q)
            return hr_arma(y_mean, p=pp, q=qq, n_iter=hr_iter, step1=step1)

        # Build valid indices
        if len(censor_rel) > 0:
            valid = np.ones(n_run, dtype=bool)
            valid[censor_rel] = False
            valid_idx = np.where(valid)[0]
        else:
            valid_idx = np.arange(n_run)

        if method == "ar":
            n_eff = len(valid_idx)
            if n_eff <= 1:
                return {"phi": np.array([], dtype=np.float64),
                        "theta": np.array([], dtype=np.float64),
                        "order": (0, 0)}

            p_cap = min(int(p_max), n_eff - 1)

            # Compute pooled ACVF from valid segments
            if len(censor_rel) > 0 and n_eff > 0:
                diffs = np.diff(valid_idx)
                seg_breaks = np.where(diffs > 1)[0]
                seg_starts_local = np.concatenate([[0], seg_breaks + 1])
                seg_ends_local = np.concatenate([seg_breaks, [len(valid_idx) - 1]])

                gamma_sum = np.zeros(p_cap + 1)
                total_contrib = np.zeros(p_cap + 1)

                for si in range(len(seg_starts_local)):
                    seg_idx = valid_idx[seg_starts_local[si] : seg_ends_local[si] + 1]
                    seg_len = len(seg_idx)
                    if seg_len > 1:
                        seg_mat = mat[seg_idx]
                        seg_pmax = min(p_cap, seg_len - 1)
                        seg_gamma = run_avg_acvf(seg_mat, seg_pmax)
                        for lag in range(seg_pmax + 1):
                            contrib = seg_len - lag
                            gamma_sum[lag] += seg_gamma[lag] * contrib
                            total_contrib[lag] += contrib

                gamma = np.where(total_contrib > 0, gamma_sum / total_contrib, 0.0)
                p_cap = max(np.max(np.where(total_contrib > 0)[0]) if np.any(total_contrib > 0) else 0, 0)
                gamma = gamma[: p_cap + 1]
            else:
                gamma = run_avg_acvf(mat[valid_idx], p_cap)

            # BIC selection
            if gamma[0] < 1e-15:
                return {"phi": np.array([], dtype=np.float64),
                        "theta": np.array([], dtype=np.float64),
                        "order": (0, 0)}

            n_eff_log = np.log(n_eff)
            sigma0 = max(gamma[0], 1e-15)
            best_bic = n_eff * np.log(sigma0) + n_eff_log
            best_phi = np.array([], dtype=np.float64)
            best_p = 0

            for pp in range(1, p_cap + 1):
                if len(gamma) < pp + 1:
                    break
                phi_try, sigma2 = levinson_durbin(gamma[: pp + 1], pp)
                sigma2 = max(sigma2, 1e-15)
                bic = n_eff * np.log(sigma2) + (pp + 1) * n_eff_log
                if bic < best_bic:
                    best_bic = bic
                    best_phi = enforce_stationary_ar(phi_try)
                    best_p = pp

            return {"phi": best_phi,
                    "theta": np.array([], dtype=np.float64),
                    "order": (best_p, 0)}
    # If p is a fixed integer (not "auto"), override BIC
    if p != "auto":
        p_fixed = int(p)
        # For fixed p, just do Yule-Walker without BIC
        def _est_run_fixed(mat, censor_rel):
            n_run = mat.shape[0]

            if method == "arma":
                if len(censor_rel) > 0:
                    valid = np.ones(n_run, dtype=bool)
                    valid[censor_rel] = False
                    y_mean = mat[valid].mean(axis=1)
                else:
                    y_mean = mat.mean(axis=1)
                from .hr_arma import hr_arma
                return hr_arma(y_mean, p=p_fixed, q=int(q), n_iter=hr_iter, step1=step1)

            if len(censor_rel) > 0:
                valid = np.ones(n_run, dtype=bool)
                valid[censor_rel] = False
                valid_idx = np.where(valid)[0]
            else:
                valid_idx = np.arange(n_run)

            if method == "ar":
                if len(valid_idx) <= p_fixed:
                    return {"phi": np.zeros(p_fixed, dtype=np.float64),
                            "theta": np.array([], dtype=np.float64),
                            "order": (p_fixed, 0)}
                gamma = run_avg_acvf(mat[valid_idx], p_fixed)
                if gamma[0] < 1e-15:
                    return {"phi": np.zeros(p_fixed, dtype=np.float64),
                            "theta": np.array([], dtype=np.float64),
                            "order": (p_fixed, 0)}
                phi_try, _ = levinson_durbin(gamma[: p_fixed + 1], p_fixed)
                phi_try = enforce_stationary_ar(phi_try)
                return {"phi": phi_try,
                        "theta": np.array([], dtype=np.float64),
                        "order": (p_fixed, 0)}
        estimates = [_est_run_fixed(m, c) for m, c in zip(run_mats, censor_by_run)]
    else:
        estimates = [_est_run(m, c) for m, c in zip(run_mats, censor_by_run)]

    # --- Pool across runs ---
    if pooling == "global":
        lens = np.array([len(rs) for rs in run_sets])
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

        phi_avg = w @ Phi
        theta_avg = w @ Th
        # Trim trailing zeros
        phi_avg = phi_avg[: pmax_len] if pmax_len > 0 else np.array([], dtype=np.float64)
        theta_avg = theta_avg[: qmax_len] if qmax_len > 0 else np.array([], dtype=np.float64)

        phi_list = [phi_avg]
        theta_list = [theta_avg]
    else:
        phi_list = [e["phi"] for e in estimates]
        theta_list = [e.get("theta", np.array([], dtype=np.float64)) for e in estimates]

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
    )
