"""AR whitening transforms.

Applies AR pre-whitening to design and data matrices so that
subsequent OLS is equivalent to GLS under the AR noise model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.signal import lfilter

from .plan import WhiteningPlan, WhitenResult


def ar_whiten(
    x: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Whiten a 1-D or 2-D array using AR coefficients.

    Applies the filter ``(1 - phi_1 L - phi_2 L^2 - ...)`` to remove
    serial correlation.

    Parameters
    ----------
    x : NDArray
        Input array.  Shape ``(n,)`` or ``(n, V)``.
    phi : NDArray
        AR coefficients, shape ``(p,)``.

    Returns
    -------
    NDArray
        Whitened array, same shape as *x* but with the first *p*
        rows replaced by their pre-whitened versions.
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    p = len(phi)

    if p == 0:
        return x.copy()

    x = np.asarray(x, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, np.newaxis]

    n = x.shape[0]
    result = x.copy()

    for t in range(p, n):
        correction = np.zeros_like(x[t])
        for k in range(p):
            correction += phi[k] * x[t - k - 1]
        result[t] = x[t] - correction

    # For the first p rows, use a simple scaling approach
    # (exact_first=False; exact approach is more complex)
    for t in range(p):
        # Scale by sqrt(1 - sum(phi^2)) as approximate correction
        scale = np.sqrt(max(1.0 - np.sum(phi[:t + 1] ** 2), 0.01))
        result[t] = x[t] * scale

    if was_1d:
        return result.ravel()
    return result


def ar_whiten_matrix(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Whiten both design and data matrices with AR coefficients.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    phi : NDArray
        AR coefficients.  Shape ``(ar_order,)`` for global or
        ``(ar_order, V)`` for voxelwise.

    Returns
    -------
    X_w, Y_w : tuple of NDArray
        Whitened design and data matrices.
    """
    phi = np.asarray(phi, dtype=np.float64)

    if phi.ndim == 1:
        # Global phi: apply same whitening to X and Y
        return ar_whiten(X, phi), ar_whiten(Y, phi)
    else:
        # Voxelwise phi: X gets whitened with mean phi,
        # Y gets whitened voxel-by-voxel
        phi_mean = np.mean(phi, axis=1)
        X_w = ar_whiten(X, phi_mean)

        Y_w = Y.copy()
        ar_order = phi.shape[0]
        n, V = Y.shape
        for v in range(V):
            phi_v = phi[:, v]
            for t in range(ar_order, n):
                correction = 0.0
                for k in range(ar_order):
                    correction += phi_v[k] * Y[t - k - 1, v]
                Y_w[t, v] = Y[t, v] - correction
            # Approximate scaling for first ar_order rows
            for t in range(ar_order):
                scale = np.sqrt(max(1.0 - np.sum(phi_v[:t + 1] ** 2), 0.01))
                Y_w[t, v] = Y[t, v] * scale

        return X_w, Y_w


def ar_covariance_matrix(
    phi: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Build the AR(p) covariance matrix (Toeplitz).

    Parameters
    ----------
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    n : int
        Matrix dimension (number of timepoints).

    Returns
    -------
    NDArray
        Covariance matrix, shape ``(n, n)``.
    """
    p = len(phi)
    if p == 0:
        return np.eye(n)

    # Compute autocorrelation from AR coefficients
    # r(0) = 1, r(k) = phi_1*r(k-1) + phi_2*r(k-2) + ...
    max_lag = n
    r = np.zeros(max_lag)
    r[0] = 1.0

    for k in range(1, max_lag):
        for j in range(min(k, p)):
            r[k] += phi[j] * r[k - j - 1]

    # Build Toeplitz matrix
    return linalg.toeplitz(r)


# ---------------------------------------------------------------------------
# Segment-aware ARMA whitening (ports fmriAR_whiten.cpp)
# ---------------------------------------------------------------------------

def arma_whiten_segments(
    y: NDArray,
    phi: NDArray,
    theta: NDArray,
    seg_starts: NDArray,
    exact_first_ar1: bool = False,
) -> NDArray:
    """Apply ARMA whitening filter with segment resets.

    At each segment boundary the filter state is reset so that
    cross-run / cross-censor-gap contamination is avoided.

    Parameters
    ----------
    y : NDArray
        Input array, shape ``(n,)`` or ``(n, V)``.
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    theta : NDArray
        MA coefficients, shape ``(q,)``.
    seg_starts : NDArray
        0-based segment start indices (must include 0).
    exact_first_ar1 : bool
        If ``True`` and ``p == 1, q == 0``, multiply the first sample
        of each segment by ``sqrt(1 - phi[0]**2)``.

    Returns
    -------
    NDArray
        Whitened array, same shape as *y*.
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)

    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, np.newaxis]

    n, v = y.shape
    p = len(phi)
    q_ord = len(theta)

    seg_starts = np.sort(np.unique(np.asarray(seg_starts, dtype=np.intp)))
    seg_ends = np.append(seg_starts[1:], n)

    out = np.empty_like(y)

    # Build filter coefficients: b = [1, -phi_1, ...], a = [1, theta_1, ...]
    b = np.concatenate([[1.0], -phi]) if p > 0 else np.array([1.0])
    a = np.concatenate([[1.0], theta]) if q_ord > 0 else np.array([1.0])

    do_exact = exact_first_ar1 and p == 1 and q_ord == 0
    exact_scale = np.sqrt(max(1.0 - phi[0] ** 2, 0.0)) if do_exact else 1.0

    for s_start, s_end in zip(seg_starts, seg_ends):
        seg = y[s_start:s_end]
        seg_len = seg.shape[0]
        if seg_len == 0:
            continue

        # Apply filter per column
        for col in range(v):
            filtered = lfilter(b, a, seg[:, col])
            out[s_start:s_end, col] = filtered

        # Exact first-sample scaling
        if do_exact:
            out[s_start] *= exact_scale

    if was_1d:
        return out.ravel()
    return out


def _sub_run_starts(n_run: int, censor_rel: NDArray) -> NDArray:
    """Compute 0-based sub-run starts from relative censor indices."""
    starts = [0]
    if len(censor_rel) > 0:
        add = censor_rel + 1
        add = add[add < n_run]
        starts = sorted(set(starts) | set(add.tolist()))
    return np.array(starts, dtype=np.intp)


def _full_run_starts(runs: NDArray, censor: Optional[NDArray], n: int) -> NDArray:
    """Merge run boundaries and censor gaps into segment starts."""
    starts = {0}
    if runs is not None:
        runs = np.asarray(runs, dtype=np.intp)
        diffs = np.where(np.diff(runs) != 0)[0] + 1
        starts.update(diffs.tolist())
    if censor is not None and len(censor) > 0:
        censor = np.asarray(censor, dtype=np.intp)
        extra = censor + 1
        extra = extra[extra < n]
        starts.update(extra.tolist())
    return np.array(sorted(starts), dtype=np.intp)


def whiten_apply(
    plan: "WhiteningPlan",
    X: NDArray,
    Y: NDArray,
    *,
    runs: Optional[NDArray] = None,
    censor: Optional[NDArray] = None,
    parcels: Optional[NDArray] = None,
) -> "WhitenResult":
    """Apply a whitening plan to design and data matrices.

    Parameters
    ----------
    plan : WhiteningPlan
        Plan from :func:`~fmrimod.ar.estimation.fit_noise`.
    X : NDArray
        Design matrix, shape ``(n, k)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    runs : NDArray, optional
        Run labels (overrides plan's runs).
    censor : NDArray, optional
        0-based censor indices (overrides plan's censor).
    parcels : NDArray, optional
        Parcel labels (overrides plan's parcels).

    Returns
    -------
    WhitenResult
    """
    from .plan import WhitenResult

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    n = X.shape[0]
    assert Y.shape[0] == n

    # Resolve runs
    if runs is None and plan.runs is not None and len(plan.runs) == n:
        runs = plan.runs
    if runs is None:
        runs = np.ones(n, dtype=np.intp)

    # Resolve censor
    if censor is None:
        censor = plan.censor

    # --- Parcel plan ---
    if plan.pooling == "parcel":
        parcels_vec = parcels if parcels is not None else plan.parcels
        if parcels_vec is None:
            raise ValueError("Parcel labels required for parcel plan")
        parcels_vec = np.asarray(parcels_vec, dtype=np.intp)

        run_starts_vec = _full_run_starts(runs, censor, n)
        phi_by = plan.phi_by_parcel or {}
        theta_by = plan.theta_by_parcel or {}

        Yw = np.empty_like(Y)
        X_by: Dict[str, NDArray] = {}

        for pid in (plan.parcel_ids or sorted(phi_by.keys())):
            key = str(pid)
            cols = np.where(parcels_vec == int(pid))[0]
            if len(cols) == 0:
                continue

            phi_v = phi_by.get(key, np.array([], dtype=np.float64))
            theta_v = theta_by.get(key, np.array([], dtype=np.float64))

            Y_sub = arma_whiten_segments(
                Y[:, cols], phi_v, theta_v, run_starts_vec,
                exact_first_ar1=plan.exact_first,
            )
            X_sub = arma_whiten_segments(
                X, phi_v, theta_v, run_starts_vec,
                exact_first_ar1=plan.exact_first,
            )
            Yw[:, cols] = Y_sub
            X_by[key] = X_sub

        return WhitenResult(X=None, Y=Yw, X_by=X_by)

    # --- Global / run plan ---
    run_labels = np.unique(runs)
    rsplits = [np.where(runs == rl)[0] for rl in run_labels]

    # Split censor by run
    censor_by_run = [np.array([], dtype=np.intp) for _ in rsplits]
    if censor is not None and len(censor) > 0:
        censor = np.asarray(censor, dtype=np.intp)
        for ri, idx in enumerate(rsplits):
            start = idx[0]
            c_in = np.intersect1d(censor, idx)
            if len(c_in):
                censor_by_run[ri] = c_in - start

    phi_list = plan.phi or []
    theta_list = plan.theta or []
    # Expand single-element lists for global pooling
    if len(phi_list) == 1:
        phi_list = phi_list * len(rsplits)
    if len(theta_list) == 1:
        theta_list = theta_list * len(rsplits)
    # Pad if needed
    while len(theta_list) < len(rsplits):
        theta_list.append(np.array([], dtype=np.float64))

    Xw_parts = []
    Yw_parts = []

    for ri, idx in enumerate(rsplits):
        Xr = X[idx]
        Yr = Y[idx]
        phi_r = phi_list[ri] if ri < len(phi_list) else np.array([], dtype=np.float64)
        theta_r = theta_list[ri] if ri < len(theta_list) else np.array([], dtype=np.float64)

        seg_starts = _sub_run_starts(len(idx), censor_by_run[ri])

        Xw_r = arma_whiten_segments(Xr, phi_r, theta_r, seg_starts,
                                     exact_first_ar1=plan.exact_first)
        Yw_r = arma_whiten_segments(Yr, phi_r, theta_r, seg_starts,
                                     exact_first_ar1=plan.exact_first)
        Xw_parts.append(Xw_r)
        Yw_parts.append(Yw_r)

    Xw = np.vstack(Xw_parts)
    Yw = np.vstack(Yw_parts)

    return WhitenResult(X=Xw, Y=Yw)


def whiten(
    X: NDArray,
    Y: NDArray,
    *,
    runs: Optional[NDArray] = None,
    censor: Optional[NDArray] = None,
    **fit_kwargs,
) -> "WhitenResult":
    """Fit noise model and apply whitening in one call.

    Convenience function: computes OLS residuals, calls
    :func:`~fmrimod.ar.estimation.fit_noise`, then
    :func:`whiten_apply`.

    Parameters
    ----------
    X, Y : NDArray
        Design and data matrices.
    runs : NDArray, optional
        Run labels.
    censor : NDArray, optional
        0-based censor indices.
    **fit_kwargs
        Additional arguments passed to ``fit_noise()``.

    Returns
    -------
    WhitenResult
    """
    from .estimation import fit_noise

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ coef
    plan = fit_noise(resid=res, runs=runs, censor=censor, **fit_kwargs)
    return whiten_apply(plan, X, Y, runs=runs, censor=censor)
