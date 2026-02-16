"""Residual diagnostics and sandwich standard errors.

Ports ``acorr.R`` and ``sandwich.R``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


def acorr_diagnostics(
    resid: NDArray,
    *,
    runs: Optional[NDArray] = None,
    max_lag: int = 20,
    aggregate: str = "mean",
) -> dict:
    """Compute residual autocorrelation for whiteness checks.

    Parameters
    ----------
    resid : NDArray
        Residual matrix, shape ``(n, V)``.
    runs : NDArray, optional
        Run labels (reserved for future per-run computation).
    max_lag : int
        Maximum lag to evaluate.
    aggregate : str
        ``"mean"``, ``"median"``, or ``"none"``.

    Returns
    -------
    dict
        ``{"lags": NDArray, "acf": NDArray, "ci": float}``
    """
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    n = resid.shape[0]
    ci = 1.96 / np.sqrt(n)

    def _acf_one(y):
        y = y - y.mean()
        var = np.sum(y ** 2)
        if var < 1e-15:
            return np.zeros(max_lag)
        acf = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            acf[lag - 1] = np.sum(y[: n - lag] * y[lag:]) / var
        return acf

    if aggregate == "none":
        A = np.column_stack([_acf_one(resid[:, j]) for j in range(resid.shape[1])])
        return {"lags": np.arange(1, max_lag + 1), "acf": A, "ci": ci}

    if aggregate == "median":
        ybar = np.median(resid, axis=1)
    else:
        ybar = resid.mean(axis=1)

    a = _acf_one(ybar)
    return {"lags": np.arange(1, max_lag + 1), "acf": a, "ci": ci}


def sandwich_from_whitened_resid(
    Xw: NDArray,
    Yw: NDArray,
    *,
    beta: Optional[NDArray] = None,
    type: str = "iid",
    df_mode: str = "rankX",
    runs: Optional[NDArray] = None,
) -> dict:
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
        ``"iid"`` or ``"hc0"`` (robust sandwich).
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

    n, p_dim = Xw.shape
    v = Yw.shape[1]

    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.inv(XtX)

    if beta is None:
        beta = XtX_inv @ (Xw.T @ Yw)

    E = Yw - Xw @ beta
    rank_X = np.linalg.matrix_rank(Xw)
    df = n - rank_X if df_mode == "rankX" else n - p_dim

    if type == "iid":
        sigma2 = np.sum(E ** 2, axis=0) / df
        se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
        return {"se": se, "sigma2": sigma2, "XtX_inv": XtX_inv, "df": df, "type": "iid"}

    # HC0 sandwich
    se = np.empty((p_dim, v))
    for j in range(v):
        e = E[:, j]
        Xe = Xw * e[:, np.newaxis]
        meat = Xe.T @ Xe
        V = XtX_inv @ meat @ XtX_inv
        se[:, j] = np.sqrt(np.diag(V))

    sigma2 = np.sum(E ** 2, axis=0) / df
    return {"se": se, "sigma2": sigma2, "XtX_inv": XtX_inv, "df": df, "type": "hc0"}
