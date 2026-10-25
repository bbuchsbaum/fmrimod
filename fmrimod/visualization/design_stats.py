"""Design diagnostics behind the design plots (no plotting here).

The numbers R fmridesign's heatmaps print: regressor correlations (across
scans or within runs), variance inflation factors with per-run intercepts,
and the standard-error multiplier of a contrast. Each has a model-level
function returning a labelled pandas object and an array-level function
the renderers and tests share.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._design_common import model_design_matrix, model_sampling_frame, timeline


def _scan_runs(model: Any, n: int) -> np.ndarray:
    return timeline(model_sampling_frame(model), n).run


def _rank(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]) if X.size else X


def correlation_matrix(X: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Column correlations; columns without variance give NaN rows/columns."""
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    X = np.asarray(X, dtype=float)
    if method == "spearman":
        X = _rank(X)
    Z = X - X.mean(axis=0)
    ss = np.sqrt((Z**2).sum(axis=0))
    flat = ss <= 1e-12 * max(1.0, float(np.abs(X).max()) if X.size else 1.0) * np.sqrt(max(len(X), 1))
    ss = np.where(flat, np.nan, ss)
    R = (Z.T @ Z) / np.outer(ss, ss)
    R[flat, :] = np.nan
    R[:, flat] = np.nan
    return np.clip(R, -1.0, 1.0)


def within_run_correlations(
    X: np.ndarray, run: np.ndarray, method: str = "pearson"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlations as the model estimates them with per-run intercepts.

    Every column is centred within each run. A column that is non-zero in
    only one run (a run-specific drift or nuisance column) is correlated on
    that run's scans only, so other runs' zeros do not dilute it; pairs of
    columns specific to different runs share no scans and are NaN (and
    ``True`` in the returned ``hidden`` mask). Returns ``(R, hidden,
    col_run)`` where ``col_run`` is the run a column is specific to (0 when
    it spans runs).
    """
    X = np.asarray(X, dtype=float).copy()
    run = np.asarray(run)
    p = X.shape[1]
    runs = np.unique(run)
    col_run = np.zeros(p, dtype=int)
    if len(runs) > 1:
        nz = X != 0
        for j in range(p):
            r = np.unique(run[nz[:, j]])
            if len(r) == 1:
                col_run[j] = int(r[0])
    for r in runs:
        rows = run == r
        X[rows] -= X[rows].mean(axis=0)
    R = correlation_matrix(X, method)
    hidden = np.zeros((p, p), dtype=bool)
    if len(runs) > 1 and (col_run > 0).any():
        for r in np.unique(col_run[col_run > 0]):
            Rr = correlation_matrix(X[run == r], method)
            idx = np.flatnonzero(col_run == r)
            R[idx, :] = Rr[idx, :]
            R[:, idx] = Rr[:, idx]
        hidden = (col_run[:, None] > 0) & (col_run[None, :] > 0) & (col_run[:, None] != col_run[None, :])
        R[hidden] = np.nan
    const = ~np.isfinite(np.diag(R))
    R[const, :] = np.nan
    R[:, const] = np.nan
    return R, hidden, col_run


def vif(X: np.ndarray, run: np.ndarray | None = None) -> np.ndarray:
    """Variance inflation factor of each column with run means removed.

    ``VIF_j = 1 / (1 - R2_j)`` from regressing column ``j`` on all other
    columns after centring every column within each run (i.e. with
    per-run intercepts in the model). Exact linear combinations of other
    columns are ``inf`` ("aliased"); columns constant within every run are
    NaN (R ``.fd_hm_vif``).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    run = np.ones(n, dtype=int) if run is None else np.asarray(run)
    Z = X.copy()
    for r in np.unique(run):
        rows = run == r
        Z[rows] -= Z[rows].mean(axis=0)
    tss = (Z**2).sum(axis=0)
    const = tss <= 1e-12 * max(1.0, float(tss.max()) if p else 1.0)
    out = np.full(p, np.nan)
    keep = np.flatnonzero(~const)
    for j in keep:
        others = keep[keep != j]
        if others.size:
            beta, *_ = np.linalg.lstsq(Z[:, others], Z[:, j], rcond=None)
            rss = float(((Z[:, j] - Z[:, others] @ beta) ** 2).sum())
        else:
            rss = float(tss[j])
        out[j] = np.inf if rss / tss[j] < 1e-8 else tss[j] / rss
    return out


def _pinv_psd(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    u, d, vt = np.linalg.svd(A)
    pos = d > tol * d.max() if d.size else d > 0
    return (vt[pos].T / d[pos]) @ u[:, pos].T


def contrast_se(X: np.ndarray, run: np.ndarray | None, W: np.ndarray) -> np.ndarray:
    """Standard-error multiplier ``sqrt(c' (X'X)^+ c)`` of each contrast.

    ``X`` is augmented with one intercept per run, so ``SE(c'b) = value x
    sigma`` for the noise SD ``sigma``. ``W`` holds one contrast per column
    (``p x k``). Contrasts outside the row space of ``X`` (not estimable)
    are NaN; all-zero contrasts are 0 (R ``.fd_hm_contrast_se``).
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    n, p = X.shape
    run = np.ones(n, dtype=int) if run is None else np.asarray(run)
    runs = np.unique(run)
    Xa = np.column_stack([X] + [(run == r).astype(float) for r in runs])
    XtX = Xa.T @ Xa
    G = _pinv_psd(XtX)
    P = XtX @ G
    Ca = np.vstack([W, np.zeros((len(runs), W.shape[1]))])
    out = np.empty(W.shape[1])
    for k in range(W.shape[1]):
        ca = Ca[:, k]
        if not np.any(ca):
            out[k] = 0.0
            continue
        resid = ca - P @ ca
        if np.linalg.norm(resid) > 1e-6 * np.linalg.norm(ca):
            out[k] = np.nan
            continue
        out[k] = float(np.sqrt(max(0.0, ca @ G @ ca)))
    return out


# --------------------------------------------------------------------------
# Model-level wrappers
# --------------------------------------------------------------------------


def _prepared(model: Any, within_run: bool | None) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, bool]:
    from ._design_common import is_baseline_model
    from .heatmaps import display_columns

    X = model_design_matrix(model)
    ci = display_columns(model)
    X = X[:, ci["col"].to_numpy(dtype=int)]
    run = _scan_runs(model, X.shape[0])
    if within_run is None:
        within_run = is_baseline_model(model)
    if within_run:
        keep = (ci["term"] != "block").to_numpy()
        X, ci = X[:, keep], ci[keep].reset_index(drop=True)
    return X, ci, run, bool(within_run)


def design_correlations(model: Any, method: str = "pearson", within_run: bool | None = False) -> pd.DataFrame:
    """Correlations between the design columns of ``model``.

    Parameters
    ----------
    model : EventModel, BaselineModel or FmriModel
    method : {"pearson", "spearman"}
    within_run : bool or None
        ``False``: across all scans with runs concatenated. ``True``: run
        intercepts dropped, columns centred within runs, run-specific
        columns correlated on their own run (pairs from different runs are
        NaN). ``None`` picks ``True`` for a baseline model.

    Returns
    -------
    pandas.DataFrame
        Square, labelled by the columns' display labels; NaN where a
        correlation is undefined.
    """
    X, ci, run, wr = _prepared(model, within_run)
    R = within_run_correlations(X, run, method)[0] if wr else correlation_matrix(X, method)
    labels = list(ci["full_label"])
    return pd.DataFrame(R, index=labels, columns=labels)


def design_vif(model: Any, within_run: bool | None = False) -> pd.Series:
    """Variance inflation factor of each design column (per-run intercepts).

    ``inf`` marks an aliased column; NaN a column constant within runs.
    With ``within_run=True`` (default for a baseline model when ``None``)
    the run-intercept columns are left out first.
    """
    X, ci, run, _ = _prepared(model, within_run)
    return pd.Series(vif(X, run), index=list(ci["full_label"]), name="vif")


__all__ = [
    "contrast_se",
    "correlation_matrix",
    "design_correlations",
    "design_vif",
    "vif",
    "within_run_correlations",
]
