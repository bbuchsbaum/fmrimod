"""Residual diagnostics and sandwich standard errors.

Ports ``acorr.R`` and ``sandwich.R``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def acorr_diagnostics(
    resid: NDArray[Any],
    *,
    runs: Optional[NDArray[Any]] = None,
    max_lag: int = 20,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Compute residual autocorrelation for whiteness checks.

    Parameters
    ----------
    resid : NDArray
        Residual matrix, shape ``(n, V)``.
    runs : NDArray, optional
        Run labels, length ``n``. When supplied, each run is centred
        separately and no lag product spans a run boundary.
    max_lag : int
        Maximum lag to evaluate.
    aggregate : str
        How the per-voxel autocorrelations are combined, lag by lag:
        ``"mean"``, ``"median"``, or ``"none"`` (a lags x voxels matrix).

    Returns
    -------
    dict
        ``{"lags": NDArray, "acf": NDArray, "ci": float}``

    Notes
    -----
    Each voxel's autocorrelation is ``sum_t y_t y_{t-k} / sum_t y_t^2``
    with the voxel centred per run (or overall), so without runs it is
    ``stats::acf``. Zero-variance voxels are NaN and skipped when
    aggregating. Aggregating per-voxel values rather than taking the
    autocorrelation of the voxel-mean series (fmriAR 617bd87) matters: the
    mean measures what the voxels share, so individually white voxels with
    a weak common signal looked strongly autocorrelated.
    """
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    n = resid.shape[0]
    ci = 1.96 / np.sqrt(n)
    if runs is None:
        seg_id = np.ones(n, dtype=np.intp)
    else:
        from .acvf import run_codes

        seg_id = run_codes(runs, n)

    _, inv = np.unique(seg_id, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    mu = np.zeros((counts.size, resid.shape[1]))
    np.add.at(mu, inv, resid)
    Rc = resid - (mu / counts[:, np.newaxis])[inv]
    den = np.sum(Rc * Rc, axis=0)
    usable = den > 0
    safe_den = np.where(usable, den, 1.0)
    A = np.full((max_lag, resid.shape[1]), np.nan)
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        ok = seg_id[k:] == seg_id[:-k]
        A[k - 1] = np.sum(Rc[k:][ok] * Rc[:-k][ok], axis=0) / safe_den
    A[:, ~usable] = np.nan

    lags = np.arange(1, max_lag + 1)
    if aggregate == "none":
        return {"lags": lags, "acf": A, "ci": ci}
    finite = np.isfinite(A)
    if aggregate == "median":
        a = np.array(
            [np.median(row[keep]) if keep.any() else np.nan
             for row, keep in zip(A, finite, strict=True)]
        )
    else:
        cnt = finite.sum(axis=1)
        a = np.where(finite, A, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        a[cnt == 0] = np.nan
    return {"lags": lags, "acf": a, "ci": ci}


def sandwich_from_whitened_resid(
    Xw: NDArray[Any],
    Yw: NDArray[Any],
    *,
    beta: Optional[NDArray[Any]] = None,
    type: str = "iid",
    df_mode: str = "rankX",
    runs: Optional[NDArray[Any]] = None,
) -> dict[str, Any]:
    """GLS standard errors from whitened residuals.

    Parameters
    ----------
    Xw : NDArray
        Whitened design matrix, shape ``(n, p)``.
    Yw : NDArray
        Whitened data matrix, shape ``(n, V)``.
    beta : NDArray, optional
        Coefficients ``(p, V)``.  Estimated if ``None``.
    type : str
        ``"iid"`` or ``"hc0"`` (robust sandwich). Coefficients that are not
        estimable because ``Xw`` is rank deficient get NaN standard errors
        and NaN rows/columns in ``XtX_inv``, as ``lm()`` reports ``NA``.
    df_mode : str
        ``"rankX"`` or ``"n-p"``.
    runs : NDArray, optional
        Reserved for future per-run scaling.

    Returns
    -------
    dict
        ``{"se": NDArray, "sigma2": NDArray, "XtX_inv": NDArray, "df": int, "type": str}``
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    Yw = np.asarray(Yw, dtype=np.float64)
    if Yw.ndim == 1:
        Yw = Yw[:, np.newaxis]
    if Xw.ndim != 2 or Xw.shape[0] != Yw.shape[0]:
        raise ValueError("Xw and Yw must have the same number of rows")
    se_type = str(type).lower()
    if se_type not in ("iid", "hc0"):
        raise ValueError(f"type must be 'iid' or 'hc0', got {type!r}")

    n, p_dim = Xw.shape
    v = Yw.shape[1]

    # Estimable columns as R's lm() finds them; aliased ones get NaN.
    est = _estimable_columns(Xw)
    rank_X = int(est.size)
    Xe = Xw[:, est]
    XtX_e = np.linalg.inv(Xe.T @ Xe) if rank_X else np.zeros((0, 0))
    XtX_inv = np.full((p_dim, p_dim), np.nan)
    XtX_inv[np.ix_(est, est)] = XtX_e

    if beta is None:
        beta_e = XtX_e @ (Xe.T @ Yw)
    else:
        beta = np.asarray(beta, dtype=np.float64)
        if beta.ndim == 1:
            beta = beta[:, np.newaxis]
        if beta.shape != (p_dim, v):
            raise ValueError(f"beta must have shape {(p_dim, v)}, got {beta.shape}")
        beta_e = beta[est]
    E = Yw - Xe @ beta_e
    df = n - rank_X if df_mode == "rankX" else n - p_dim
    sigma2 = np.sum(E**2, axis=0) / df

    se = np.full((p_dim, v), np.nan)
    if se_type == "iid":
        se[est] = np.sqrt(np.outer(np.diag(XtX_e), sigma2))
    else:
        # HC0: Var(beta_k) = sum_t H_kt^2 e_t^2 with H = (X'X)^-1 X'.
        H = XtX_e @ Xe.T
        se[est] = np.sqrt((H * H) @ (E * E))
    return {"se": se, "sigma2": sigma2, "XtX_inv": XtX_inv, "df": df, "type": se_type}


def _estimable_columns(X: NDArray[np.float64], tol: float = 1e-7) -> NDArray[np.intp]:
    """Columns R's ``qr()`` (LINPACK ``dqrdc2``) keeps, in their original order.

    A column is aliased when its norm after projection on the columns kept
    before it falls below ``tol`` times its own norm. Unlike full column
    pivoting, this drops the *later* of two collinear columns, so the
    non-estimable coefficient is the one ``lm()`` reports as ``NA``.
    """
    n, p = X.shape
    Q = np.zeros((n, 0))
    keep: list[int] = []
    for j in range(p):
        col = X[:, j]
        norm0 = float(np.linalg.norm(col))
        if norm0 == 0.0 or not np.isfinite(norm0):
            continue
        r = col - Q @ (Q.T @ col)
        r = r - Q @ (Q.T @ r)  # re-orthogonalise
        rn = float(np.linalg.norm(r))
        if rn > tol * norm0:
            keep.append(j)
            Q = np.column_stack([Q, r / rn])
    return np.array(keep, dtype=np.intp)
