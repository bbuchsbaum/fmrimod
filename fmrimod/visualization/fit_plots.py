"""Plots of a fitted GLM and of its model's regressors.

Ports R fmrireg's ``R/plot-fmri_lm.R`` (``autoplot.fmri_lm``) and
``plot.fmri_model`` onto fmrimod's design system. Five views of a fit:

- ``"betas"`` / ``"contrasts"`` - with no voxel, the distribution of t
  across voxels per regressor (a deterministic sina plot) with the
  p < .001 critical lines and the share of voxels beyond them; with a
  voxel, a coefficient plot with confidence intervals.
- ``"hrf"`` - the estimated response per condition, the term's HRF basis
  weighted by the coefficients, with a pointwise confidence band.
- ``"timecourse"`` - observed and fitted signal, onsets and residuals for
  one voxel.
- ``"residuals"`` - residual autocorrelation against the white-noise range
  this design induces, with a generalised Ljung-Box test.

Every number a figure shows is computed by a small pure function here
(:func:`critical_t`, :func:`sina_offsets`, :func:`share_beyond`,
:func:`coefficient_intervals`, :func:`hrf_curves`, :func:`voxel_refit`,
:func:`acf_cols`, :func:`acf_null_cov`, :func:`residual_acf`,
:func:`portmanteau`), so the renderers only draw.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._helpers import (
    DOT,
    MINUS,
    axis_number_formatter,
    budget_yticks,
    column_info,
    finish,
    freeze_fonts,
    halo_text,
    new_figure,
    num,
    row_label,
    run_axis,
    themed,
    time_axis,
)
from .style import BASE_SIZE, CATEGORICAL, INK, colour_map, theme_rc, tints

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.transforms import Transform

    from fmrimod.glm.fmri_lm import FmriLm
    from fmrimod.model.fmri_model import FmriModel

FloatArray = NDArray[np.float64]
VoxelArg = int | Sequence[int] | NDArray[np.integer] | None
Size = tuple[float, float] | None

KINDS = ("betas", "contrasts", "hrf", "timecourse", "residuals")
NULL_P = 0.001
MAX_HRF_LINES = 200
MAX_ACF_VOXELS = 200
BIN_ABOVE = 5000
GOLDEN = 0.6180339887


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def _pct(x: float) -> str:
    """``0%``, ``<1%`` or a whole percent (R ``.pct``)."""
    if not np.isfinite(x):
        return ""
    if x == 0:
        return "0%"
    if x < 0.01:
        return "<1%"
    return f"{100 * x:.0f}%"


def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


def _signed(x: float, fmt: str) -> str:
    return format(x, fmt).replace("-", MINUS)


def _df_text(d: Sequence[float] | float) -> str:
    """One df, or a range when voxels differ (R ``.df_text``)."""
    arr = np.atleast_1d(np.asarray(d, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "Inf"
    lo, hi = np.round(arr.min(), 1), np.round(arr.max(), 1)
    if hi - lo < 0.5:
        return f"{round(lo):.0f}"
    return f"{lo:g}–{hi:g}"


def _join_notes(notes: Sequence[str | None]) -> str | None:
    kept = [n.strip() for n in notes if n and n.strip()]
    return "\n".join(kept) if kept else None


# --------------------------------------------------------------------------
# Fit facts
# --------------------------------------------------------------------------


def _model(fit: FmriLm | FmriModel) -> FmriModel:
    return getattr(fit, "model", fit)


def _event_cols(fit: FmriLm) -> NDArray[np.intp]:
    return np.asarray(_model(fit).event_column_indices, dtype=np.intp)


def _run_lengths(model: FmriModel) -> list[int]:
    try:
        lens = [int(v) for v in model.n_timepoints]
    except Exception:  # pragma: no cover - models without run structure
        lens = [int(np.asarray(model.design_matrix_array()).shape[0])]
    return lens


def _scan_times(model: FmriModel, n: int) -> FloatArray:
    sf = getattr(model, "sampling_frame", None)
    if sf is not None:
        try:
            t = np.asarray(sf.samples, dtype=float)
            if t.size == n:
                return t
        except Exception:  # pragma: no cover - frame without samples
            pass
    return np.arange(n, dtype=float)


def _tr_text(model: FmriModel) -> str | None:
    sf = getattr(model, "sampling_frame", None)
    if sf is None:
        return None
    try:
        trs = np.unique(np.atleast_1d(np.asarray(sf.TR, dtype=float)))
    except Exception:  # pragma: no cover
        return None
    return "/".join(f"{v:g}" for v in trs)


def _is_runwise(fit: FmriLm) -> bool:
    """Whether the fit estimated coefficients run by run (the default)."""
    n_runs = len(_run_lengths(_model(fit)))
    if n_runs < 2:
        return False
    solver = str(getattr(getattr(fit, "provenance", None), "solver_path", "") or "")
    if solver:
        return solver in ("RunwiseEngine", "ChunkwiseEngine")
    return True


def _strategy_text(fit: FmriLm) -> str:
    n_runs = len(_run_lengths(_model(fit)))
    if _is_runwise(fit):
        return f"Runwise fit: coefficients estimated per run and pooled across {n_runs} runs."
    return "Fit jointly across all scans."


def _inference_df(fit: FmriLm) -> float:
    method = getattr(fit, "_inference_df", None)
    d = method() if callable(method) else getattr(fit, "residual_df", np.inf)
    return float(d)


def _check_voxels(fit: FmriLm, voxel: VoxelArg, single: bool = False) -> list[int]:
    nvox = int(np.asarray(fit.betas).shape[1])
    vals = np.atleast_1d(np.asarray(voxel))
    ok = (
        vals.size >= 1
        and np.issubdtype(vals.dtype, np.number)
        and np.all(np.isfinite(vals))
        and np.all(vals == np.round(vals))
        and np.all((vals >= 0) & (vals < nvox))
    )
    if not ok:
        raise ValueError(f"`voxel` must contain integer indices between 0 and {nvox - 1}.")
    out = list(dict.fromkeys(int(v) for v in vals))
    if single and len(out) != 1:
        raise ValueError("`voxel` must be a single index for this view.")
    return out


def _unidentified(X: FloatArray, tol: float = 1e-7) -> NDArray[np.bool_]:
    """Columns whose coefficient the design does not identify.

    Coefficient ``j`` is estimable when the unit vector ``e_j`` lies in the
    row space of ``X``, i.e. every null-space vector of ``X`` is zero in
    position ``j``. All-zero columns are not estimable.
    """
    out = np.ones(X.shape[1], dtype=bool)
    live = (X != 0).any(axis=0)
    if not live.any():
        return out
    _, sv, vt = np.linalg.svd(X[:, live], full_matrices=True)
    rank = int(np.sum(sv > tol * sv[0])) if sv.size and sv[0] > 0 else 0
    null = vt[rank:].T
    out[live] = np.linalg.norm(null, axis=1) > 1e-6 if null.size else False
    return out


def aliased_columns(fit: FmriLm) -> NDArray[np.bool_]:
    """Design columns whose coefficient is not estimable.

    R fmrireg reports such coefficients as ``NA``; fmrimod's pseudoinverse
    solve returns finite minimum-norm values for them instead, so the plots
    find them from the design: a column is aliased when the design does not
    identify its coefficient (per run for runwise fits, in any run in which
    the column is non-zero; jointly otherwise). Unlike R's ``lm``, which
    keeps the earlier column of a dependent set, every column of the set is
    marked, since the pseudoinverse splits the effect among them.
    """
    model = _model(fit)
    X = np.asarray(model.design_matrix_array(), dtype=float)
    lens = _run_lengths(model)
    if not _is_runwise(fit) or sum(lens) != X.shape[0]:
        return _unidentified(X)
    starts = np.concatenate([[0], np.cumsum(lens)])
    alias = np.zeros(X.shape[1], dtype=bool)
    ever = np.zeros(X.shape[1], dtype=bool)
    for r in range(len(lens)):
        Xr = X[starts[r]:starts[r + 1]]
        live = (Xr != 0).any(axis=0)
        alias |= live & _unidentified(Xr)
        ever |= live
    return alias | ~ever


def _masked(values: FloatArray, alias: NDArray[np.bool_]) -> FloatArray:
    out = np.array(values, dtype=float, copy=True)
    out[alias] = np.nan
    return out


def peak_voxel(fit: FmriLm) -> int:
    """Voxel with the largest absolute event t-statistic (R ``.peak_voxel``)."""
    t = _masked(fit.tstat(), aliased_columns(fit))[_event_cols(fit)]
    with np.errstate(invalid="ignore"):
        a = np.abs(t)
    a[~np.isfinite(a)] = -np.inf
    score = a.max(axis=0) if a.size else np.zeros(t.shape[1])
    return int(np.argmax(score))


def _event_info(fit: FmriLm) -> pd.DataFrame:
    info = column_info(fit)
    return info[info["kind"] == "event"].reset_index(drop=True)


def condition_colours(info: pd.DataFrame) -> dict[str, str]:
    """Colour per event condition label, in design order (R ``.condition_colours``)."""
    ev = info[info["kind"] == "event"]
    return colour_map(list(ev["label"]))


# --------------------------------------------------------------------------
# Pure computations: estimates view
# --------------------------------------------------------------------------


def critical_t(df: float, p: float = NULL_P) -> float:
    """Two-sided critical t at level ``p``."""
    from scipy import stats as sp_stats

    return float(sp_stats.t.isf(p / 2.0, df))


def _bw_nrd0(x: FloatArray) -> float:
    """Silverman's rule of thumb (R ``bw.nrd0``)."""
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    lo = min(sd, (q75 - q25) / 1.34)
    if lo <= 0:
        lo = sd or abs(float(x[0])) or 1.0
    return 0.9 * lo * x.size ** (-0.2)


def sina_offsets(x: Sequence[float], width: float = 0.38) -> FloatArray:
    """Vertical offsets proportional to the local density (R ``.sina_offset``).

    A Gaussian kernel density (Silverman bandwidth, 512-point grid as R's
    ``density``) scales a deterministic golden-ratio sequence assigned by
    rank, so the same data always give the same picture.
    """
    xs = np.asarray(x, dtype=float)
    n = xs.size
    if n < 3 or float(np.std(xs)) == 0.0:
        return np.zeros(n)
    bw = _bw_nrd0(xs)
    grid = np.linspace(xs.min() - 3 * bw, xs.max() + 3 * bw, 512)
    dens_grid = np.zeros_like(grid)
    for start in range(0, n, 2048):
        chunk = xs[start : start + 2048]
        dens_grid += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(axis=1)
    dens = np.interp(xs, grid, dens_grid)
    rank = np.argsort(np.argsort(xs, kind="stable"), kind="stable")
    u = (np.arange(1, n + 1) * GOLDEN) % 1.0
    return (u[rank] - 0.5) * 2.0 * width * dens / dens.max()


def share_beyond(t: FloatArray, crit: float) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Share of voxels with ``t > crit`` and ``t < -crit`` per column.

    ``t`` is ``(voxels, k)``. Non-finite values are ignored; a column with
    no finite value is not estimable.
    """
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(t)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(n > 0, ((t > crit) & finite).sum(axis=0) / np.maximum(n, 1), np.nan)
        neg = np.where(n > 0, ((t < -crit) & finite).sum(axis=0) / np.maximum(n, 1), np.nan)
    return pos, neg, n > 0


def coefficient_intervals(
    est: FloatArray, se: FloatArray, t: FloatArray, df: FloatArray | float, level: float
) -> pd.DataFrame:
    """Estimate ± t_crit·SE, two-sided p and the R annotation text."""
    from scipy import stats as sp_stats

    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    t = np.asarray(t, dtype=float)
    d = np.broadcast_to(np.asarray(df, dtype=float), est.shape)
    q = sp_stats.t.isf((1 - level) / 2.0, d)
    p = 2.0 * sp_stats.t.sf(np.abs(t), d)
    txt = [
        f"t = {_signed(tv, '.1f')}, {_p_text(pv)}" if np.isfinite(tv) else "not estimable (aliased)"
        for tv, pv in zip(t, p, strict=True)
    ]
    return pd.DataFrame(
        {"est": est, "se": se, "t": t, "df": d, "lo": est - q * se, "hi": est + q * se, "p": p, "txt": txt}
    )


# --------------------------------------------------------------------------
# Pure computations: HRF view
# --------------------------------------------------------------------------


def hrf_terms(fit: FmriLm, sample_at: Sequence[float]) -> list[dict[str, Any]]:
    """Each event term's HRF basis at ``sample_at`` and its coefficient blocks.

    Mirrors :func:`fmrimod.accessors.fitted_hrf`: ``G`` is the term's HRF
    (one column per basis function); ``blocks`` maps each condition label
    to that condition's design-column positions, ordered by basis.
    """
    model = _model(fit)
    em = getattr(model, "event_model", None)
    if em is None:
        return []
    sample = np.asarray(sample_at, dtype=float)
    info = _event_info(fit)
    col_idx = getattr(em, "column_indices", {}) or {}
    out = []
    for term in getattr(em, "terms", []):
        name = getattr(term, "name", None)
        spec = getattr(term, "hrf", None)
        idx = list(col_idx.get(name, []))
        if spec is None or not idx:
            continue
        hrf_obj = em._resolve_hrf(spec)
        G = np.asarray(hrf_obj(sample), dtype=float)
        G = G.reshape(len(sample), -1) if G.ndim == 1 else G
        if G.shape[0] != len(sample):
            G = G.T
        nb = G.shape[1]
        if len(idx) % nb:
            continue
        sub = info.iloc[idx]
        blocks: dict[str, NDArray[np.intp]] = {}
        for lab, grp in sub.groupby("label", sort=False):
            order = np.argsort(pd.to_numeric(grp["basis_ix"], errors="coerce").fillna(1).to_numpy(), kind="stable")
            blocks[str(lab)] = grp["col"].to_numpy()[order].astype(np.intp)
        if any(len(b) != nb for b in blocks.values()):
            continue
        term_label = str(sub["term_label"].iloc[0] or name)
        out.append(
            {
                "term": str(name),
                "term_label": term_label,
                "G": G,
                "blocks": blocks,
                "hrf_name": str(getattr(hrf_obj, "name", "HRF")),
            }
        )
    return out


def _coef_cov(fit: FmriLm, voxel: int, idx: NDArray[np.intp]) -> FloatArray:
    """``sigma² (X'X)^{-1}`` block for one voxel (its own matrix for voxelwise AR)."""
    xtx = np.asarray(fit.XtXinv, dtype=float)
    block = xtx[voxel][np.ix_(idx, idx)] if xtx.ndim == 3 else xtx[np.ix_(idx, idx)]
    s = float(np.asarray(fit.sigma, dtype=float)[voxel])
    return s * s * block


def hrf_curves(
    fit: FmriLm,
    voxels: Sequence[int],
    sample_at: Sequence[float],
    level: float = 0.95,
    max_lines: int = MAX_HRF_LINES,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    """Estimated responses with pointwise intervals.

    One voxel: the band is ``t_{df} · sqrt(g' V g)`` with ``V`` the
    coefficient covariance ``sigma² (X'X)^{-1}``. Several voxels: the mean
    response across voxels and a t interval of that mean (voxels treated as
    independent), plus up to ``max_lines`` per-voxel curves.

    Returns the curve table (``term, condition, time, mean, lo, hi, sig,
    nbasis, peak_t, peak_v``), the per-voxel lines, the conditions that are
    not estimable, and the term descriptions from :func:`hrf_terms`.
    """
    from scipy import stats as sp_stats

    sample = np.asarray(sample_at, dtype=float)
    terms = hrf_terms(fit, sample)
    voxels = list(voxels)
    beta = _masked(fit.betas, aliased_columns(fit))
    single = len(voxels) == 1
    df = _inference_df(fit)
    if len(voxels) > max_lines:
        keep = np.unique(np.round(np.linspace(0, len(voxels) - 1, max_lines)).astype(int))
    else:
        keep = np.arange(len(voxels))
    curves, lines, aliased = [], [], []
    for td in terms:
        G = td["G"]
        for cond, idx in td["blocks"].items():
            B = beta[np.ix_(idx, voxels)]  # basis x voxels
            if not np.isfinite(B).any():
                aliased.append(cond)
                continue
            B = np.where(np.isfinite(B), B, 0.0)
            H = G @ B  # time x voxels
            if single:
                m = H[:, 0]
                half = np.full_like(m, np.nan)
                V = _coef_cov(fit, voxels[0], idx)
                if np.all(np.isfinite(V)):
                    q = sp_stats.t.isf((1 - level) / 2.0, df)
                    half = q * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", G, V, G), 0.0))
            else:
                m = H.mean(axis=1)
                q = sp_stats.t.isf((1 - level) / 2.0, H.shape[1] - 1)
                half = q * H.std(axis=1, ddof=1) / math.sqrt(H.shape[1])
                for k in keep:
                    lines.append(
                        pd.DataFrame(
                            {"term": td["term"], "condition": cond, "time": sample,
                             "value": H[:, k], "voxel": voxels[k]}
                        )
                    )
            pk = int(np.argmax(np.abs(m)))
            lo, hi = m - half, m + half
            sig = True if np.all(np.isnan(half)) else bool(lo[pk] > 0 or hi[pk] < 0)
            curves.append(
                pd.DataFrame(
                    {"term": td["term"], "condition": cond, "time": sample, "mean": m, "lo": lo,
                     "hi": hi, "sig": sig, "nbasis": G.shape[1], "peak_t": sample[pk],
                     "peak_v": m[pk]}
                )
            )
    cdf = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
        columns=["term", "condition", "time", "mean", "lo", "hi", "sig", "nbasis", "peak_t", "peak_v"])
    ldf = pd.concat(lines, ignore_index=True) if lines else pd.DataFrame(
        columns=["term", "condition", "time", "value", "voxel"])
    return cdf, ldf, list(dict.fromkeys(aliased)), terms


# --------------------------------------------------------------------------
# Pure computations: time course and residuals
# --------------------------------------------------------------------------


def _lstsq_fitted(X: FloatArray, y: FloatArray) -> FloatArray:
    if X.shape[1] == 0:
        return np.zeros_like(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ coef


def voxel_refit(
    y: FloatArray, Xe: FloatArray, Xb: FloatArray, run_id: Sequence[int], runwise: bool
) -> dict[str, Any]:
    """OLS fit of the full design to one voxel (R ``.voxel_refit``).

    Run by run when ``runwise`` (baseline columns that are zero in a run
    are dropped there), jointly otherwise. ``partial_r2`` is
    ``1 - RSS_full / RSS_baseline``: the share of the variance the baseline
    leaves that the event regressors explain (``nan`` when the baseline
    fits perfectly).
    """
    y = np.asarray(y, dtype=float)
    run_id = np.asarray(run_id)
    n = y.size
    fitted = np.full(n, np.nan)
    base = np.full(n, np.nan)
    pieces = [np.flatnonzero(run_id == r) for r in dict.fromkeys(run_id.tolist())] if runwise else [np.arange(n)]
    for rows in pieces:
        xb = Xb[rows]
        xb = xb[:, (xb != 0).any(axis=0)]
        full = np.hstack([Xe[rows], xb])
        fitted[rows] = _lstsq_fitted(full, y[rows])
        base[rows] = _lstsq_fitted(xb, y[rows]) if xb.shape[1] else np.mean(y[rows])
    rss_full = float(np.sum((y - fitted) ** 2))
    rss_base = float(np.sum((y - base) ** 2))
    tiny = 1e-10 * max(float(np.sum(y**2)), np.finfo(float).tiny)
    r2 = np.nan if rss_base <= tiny else 1.0 - rss_full / rss_base
    return {"fitted": fitted, "resid": y - fitted, "partial_r2": r2}


def acf_cols(E: FloatArray, lag_max: int) -> FloatArray:
    """Autocorrelation at lags ``1..lag_max`` of each column, ``(lags, voxels)``."""
    E = np.asarray(E, dtype=float)
    E = E - E.mean(axis=0)
    den = np.sum(E**2, axis=0)
    den = np.where(den == 0, np.nan, den)
    n = E.shape[0]
    out = np.full((lag_max, E.shape[1]), np.nan)
    for k in range(1, lag_max + 1):
        if k < n:
            out[k - 1] = np.sum(E[: n - k] * E[k:], axis=0) / den
    return out


def ar_whiten(E: FloatArray, phi: Sequence[float]) -> FloatArray:
    """AR(p) prewhitening filter; drops the first ``p`` rows (R ``.ar_whiten``)."""
    E = np.asarray(E, dtype=float)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = phi.size
    n = E.shape[0]
    if p == 0 or n <= p + 2:
        return E
    out = E.copy()
    for k in range(1, p + 1):
        out[p:] -= phi[k - 1] * E[p - k : n - k]
    return out[p:]


def acf_null_cov(Qs: FloatArray, lag_max: int) -> FloatArray:
    """Null covariance of residual autocorrelations at lags ``1..lag_max``.

    For residuals of white noise with residual maker ``M = I - Q Q'`` (``Q``
    an orthonormal basis of the run's design columns), ``r_k ≈ e' A_k e /
    tr(M)`` with ``A_k`` the symmetrised lag-k shift, so ``Cov(r_j, r_k) =
    2 tr(A_j M A_k M) / tr(M)²``. Expanding ``M`` needs only ``n x p`` and
    ``p x p`` products (R ``.acf_null_cov``).
    """
    Qs = np.asarray(Qs, dtype=float)
    n, p = Qs.shape
    tr_m = n - float(np.sum(Qs**2))
    AQ = []
    for k in range(1, lag_max + 1):
        if k >= n:
            AQ.append(np.zeros((n, p)))
            continue
        up = np.vstack([Qs[k:], np.zeros((k, p))])
        down = np.vstack([np.zeros((k, p)), Qs[: n - k]])
        AQ.append((up + down) / 2.0)
    G = [Qs.T @ a for a in AQ]
    S = np.zeros((lag_max, lag_max))
    for j in range(lag_max):
        for k in range(j, lag_max):
            tr = (max(n - (j + 1), 0) / 2.0 if j == k else 0.0) - 2.0 * float(np.sum(AQ[j] * AQ[k]))
            tr += float(np.sum(G[j] * G[k].T))
            S[j, k] = S[k, j] = 2.0 * tr / tr_m**2
    return S


def _orth_basis(x: FloatArray) -> FloatArray:
    """Orthonormal basis of the column space (pivoted QR, R ``qr`` tolerance)."""
    from scipy.linalg import qr

    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 0))
    q, r, _ = qr(x, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int(np.sum(d > 1e-7 * d[0])) if d.size and d[0] > 0 else 0
    return q[:, :rank]


def residual_acf(
    Y: FloatArray,
    X: FloatArray,
    run_id: Sequence[int],
    runwise: bool,
    lag_max: int,
    phi: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Residual autocorrelation of several voxels and its white-noise reference.

    Fits the full design by least squares (per run when ``runwise``, jointly
    otherwise) and pools the autocorrelation of the residuals at lags
    ``1..lag_max`` across runs by run length. With ``phi`` (AR coefficients
    per run) data and design are first prewhitened run by run (GLS
    residuals). ``expect`` is the white-noise expectation
    ``-Σ_t (QQ')[t, t+k] / tr(M)`` and ``cov`` the exact null covariance
    (:func:`acf_null_cov`), pooled the same way (R ``.residual_acf``).
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    run_id = np.asarray(run_id)
    run_keys = list(dict.fromkeys(run_id.tolist()))
    Yw, Xw = [], []
    for r, key in enumerate(run_keys):
        rows = np.flatnonzero(run_id == key)
        y, x = Y[rows], X[rows]
        if phi is not None:
            y = ar_whiten(y, phi[r])
            x = ar_whiten(x, phi[r])
        Yw.append(y)
        Xw.append(x)
    groups = [[r] for r in range(len(run_keys))] if runwise else [list(range(len(run_keys)))]
    acf_sum: FloatArray | float = 0.0
    exp_sum: FloatArray | float = 0.0
    cov_sum: FloatArray | float = 0.0
    n_tot = 0
    for g in groups:
        y = np.vstack([Yw[r] for r in g])
        x = np.vstack([Xw[r] for r in g])
        x = x[:, (x != 0).any(axis=0)]
        seg = np.concatenate([np.full(Yw[r].shape[0], i) for i, r in enumerate(g)])
        Q = _orth_basis(x)
        E = y - Q @ (Q.T @ y)
        for s in range(len(g)):
            rows = np.flatnonzero(seg == s)
            n = rows.size
            Qs = Q[rows]
            tr_m = n - float(np.sum(Qs**2))
            e_k = np.array(
                [
                    -float(np.sum(Qs[: n - k] * Qs[k:])) / tr_m if k < n else 0.0
                    for k in range(1, lag_max + 1)
                ]
            )
            acf_sum = acf_sum + n * acf_cols(E[rows], lag_max)
            exp_sum = exp_sum + n * e_k
            cov_sum = cov_sum + n**2 * acf_null_cov(Qs, lag_max)
            n_tot += n
    return {
        "acf": np.asarray(acf_sum) / n_tot,
        "expect": np.asarray(exp_sum) / n_tot,
        "cov": np.asarray(cov_sum) / n_tot**2,
        "n": n_tot,
    }


def portmanteau(acf: FloatArray, expect: FloatArray, cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Generalised Ljung-Box statistic and p-value per voxel.

    ``Q = d' S^{-1} d`` with ``d`` the deviation of the residual
    autocorrelations from their design-induced expectation and ``S`` their
    exact null covariance, referred to chi-square on ``lag_max`` df (the AR
    coefficients are pooled, not fitted per series, so none are subtracted).
    """
    from scipy import stats as sp_stats

    acf = np.asarray(acf, dtype=float).reshape(len(expect), -1)
    d = acf - np.asarray(expect, dtype=float)[:, None]
    cov = np.asarray(cov, dtype=float)
    try:
        q = np.sum(d * np.linalg.solve(cov, d), axis=0)
    except np.linalg.LinAlgError:
        q = np.sum(d**2 / np.diag(cov)[:, None], axis=0)
    return q, sp_stats.chi2.sf(q, df=len(expect))


def acf_band(expect: FloatArray, cov: FloatArray, level: float = 0.95) -> tuple[FloatArray, FloatArray]:
    """Pointwise white-noise range ``expect ± z · sqrt(diag(cov))``."""
    from scipy import stats as sp_stats

    half = sp_stats.norm.isf((1 - level) / 2.0) * np.sqrt(np.maximum(np.diag(cov), 0.0))
    return expect - half, expect + half


def _fit_ar_phi(fit: FmriLm, n_runs: int) -> tuple[list[FloatArray] | None, bool]:
    """AR coefficients the fit applied, one vector per run, and whether they
    were pooled across voxels from a voxelwise fit (median)."""
    a = getattr(fit, "ar_params", None)
    if a is None or getattr(fit, "ma_params", None) is not None:
        return None, False
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None, False
    ar_cfg = getattr(getattr(fit, "config", None), "ar", None)
    voxelwise = bool(getattr(ar_cfg, "voxelwise", False))
    if voxelwise:
        med = np.nanmedian(a, axis=-1)  # drop the voxel axis
        per_run = [med] * n_runs if med.ndim == 1 else [med[r] for r in range(med.shape[0])]
    else:
        per_run = [a] * n_runs if a.ndim == 1 else [a[r] for r in range(a.shape[0])]
    if len(per_run) != n_runs or any(not np.all(np.isfinite(p)) for p in per_run):
        return None, voxelwise
    return [np.atleast_1d(p) for p in per_run], voxelwise


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def _voxel_data(model: FmriModel, voxels: Sequence[int], what: str) -> FloatArray:
    from ..accessors import get_data_matrix

    dataset = getattr(model, "dataset", None)
    if dataset is None:
        raise ValueError(f"This fit's model does not carry its dataset; {what} cannot be computed.")
    try:
        Y = np.asarray(get_data_matrix(dataset), dtype=float)
    except Exception as exc:
        raise ValueError(f"This fit's dataset could not be read; {what} cannot be computed ({exc}).") from exc
    n = sum(_run_lengths(model))
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(
            f"The dataset has {Y.shape[0] if Y.ndim else 0} scans but the design has {n}; "
            f"{what} cannot be computed."
        )
    return Y[:, list(voxels)]


def _design_parts(model: FmriModel) -> tuple[FloatArray, FloatArray]:
    X = np.asarray(model.design_matrix_array(), dtype=float)
    ev = np.asarray(model.event_column_indices, dtype=np.intp)
    bl = np.asarray(model.baseline_column_indices, dtype=np.intp)
    return X[:, ev], X[:, bl]


def _event_onsets(model: FmriModel, info: pd.DataFrame) -> pd.DataFrame | None:
    """Global onset and condition label of every event (R ``.event_onsets``).

    Labels match :func:`column_info`, so onsets share the condition colours.
    Parametric terms share onsets with the factor term; one tick per onset.
    """
    em = getattr(model, "event_model", None)
    if em is None or not hasattr(em, "_create_event_terms"):
        return None
    lens = _run_lengths(model)
    sf = getattr(model, "sampling_frame", None)
    trs = np.broadcast_to(np.atleast_1d(np.asarray(getattr(sf, "TR", 1.0), dtype=float)), (len(lens),))
    offsets = np.concatenate([[0.0], np.cumsum(np.asarray(lens) * trs)[:-1]])
    raw_ids = getattr(em, "blockids", None)
    blockids = np.asarray(raw_ids if raw_ids is not None else [], dtype=float)
    ev_info = info[info["kind"] == "event"]
    rows = []
    order: list[str] = []
    for et in em._create_event_terms():
        sub = ev_info[ev_info["term"] == et.name]
        labels = list(dict.fromkeys(sub["label"]))
        onsets = np.asarray(et.events[0].onsets, dtype=float)
        if onsets.size == 0:
            continue
        cats = [e for e in et.events if getattr(e, "event_type", "") == "categorical"]
        if cats and len(cats) == len(et.events):
            vals = [np.asarray(getattr(e, "_values", getattr(e, "values", [])), dtype=object) for e in cats]
            conds = [" × ".join(str(v[i]) for v in vals) for i in range(onsets.size)]
            prefix = {c.split(": ", 1)[-1]: c for c in labels}
            conds = [prefix.get(c, c) for c in conds]
        else:
            conds = [labels[0] if labels else str(et.name)] * onsets.size
        bid = blockids if blockids.size == onsets.size else np.ones(onsets.size)
        glob = onsets + offsets[np.clip(bid.astype(int) - 1, 0, len(lens) - 1)]
        order.extend(labels or [str(et.name)])
        rows.append(pd.DataFrame({"onset": glob, "condition": conds}))
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True).drop_duplicates("onset", keep="first")
    levels = [c for c in dict.fromkeys(order) if c in set(out["condition"])]
    levels += [c for c in dict.fromkeys(out["condition"]) if c not in levels]
    out["condition"] = pd.Categorical(out["condition"], categories=levels)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def plot_fit(
    fit: FmriLm,
    kind: str = "betas",
    voxel: VoxelArg = None,
    level: float = 0.95,
    sample_at: Sequence[float] | None = None,
    direct_labels: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a fitted GLM (R fmrireg ``autoplot.fmri_lm``).

    Parameters
    ----------
    fit : FmriLm
        The fitted model.
    kind : {"betas", "contrasts", "hrf", "timecourse", "residuals"}
        ``"betas"`` (alias ``"estimates"``) and ``"contrasts"``: with no
        ``voxel``, the distribution of t across voxels per event regressor
        (or t-contrast) with dashed lines at the two-sided p < .001 critical
        value and, in the right margin, the share of voxels beyond each;
        with ``voxel``, a coefficient plot of estimates with ``level``
        confidence intervals, one panel per event term. ``"hrf"``: the
        estimated response per condition (the term's HRF basis weighted by
        the coefficients) with a pointwise confidence band; a curve whose
        band covers 0 at its peak is dashed and marked "(n.s.)". For
        several voxels the mean response, a t interval of the mean and
        faint per-voxel curves. ``"timecourse"``: observed and fitted
        signal, onsets per condition and residuals for one voxel.
        ``"residuals"``: residual autocorrelation at lags 1 to 10 against
        the white-noise range this design induces, OLS and, for AR fits,
        prewhitened.
    voxel : int or sequence of int, optional
        0-based voxel index (indices for ``"betas"``, ``"contrasts"``,
        ``"hrf"`` and ``"residuals"``). ``None``: all voxels for the
        estimates views, the voxel with the largest |t| for ``"hrf"`` and
        ``"timecourse"``, up to 200 evenly spaced voxels for
        ``"residuals"``.
    level : float
        Confidence level of intervals and bands.
    sample_at : sequence of float, optional
        Times (s) at which to evaluate the HRF; default ``0, 0.25, ..., 20``.
    direct_labels : bool
        ``"hrf"``: label each curve beside its peak (default) or use a
        legend.
    title, subtitle, caption : str, optional
        ``None`` uses the default, ``""`` suppresses it.
    figsize : (float, float), optional
        Figure size in inches; scaled to the content by default.
    ax : matplotlib Axes, optional
        Draw a single-panel view into this axes.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Critical values and intervals use the fit's inference df (the
    effective df of a robust fit). The time-course and residual views refit
    the full design to the voxel by OLS (run by run for runwise fits,
    jointly otherwise) from the model's dataset, since a fit keeps no
    residuals; for AR or robust fits these are the unweighted counterparts.
    """
    kind = str(kind)
    if kind == "estimates":
        kind = "betas"
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}; got {kind!r}")
    if not (isinstance(level, (int, float)) and np.isfinite(level) and 0 < level < 1):
        raise ValueError("`level` must be a single number between 0 and 1.")
    if not isinstance(direct_labels, (bool, np.bool_)):
        raise ValueError("`direct_labels` must be True or False.")
    if sample_at is None:
        sample_at = np.arange(0.0, 20.25, 0.25)
    text = {"title": title, "subtitle": subtitle, "caption": caption}
    if kind in ("betas", "contrasts"):
        return _plot_stats(fit, kind, voxel, level, figsize, ax, text)
    if kind == "hrf":
        return _plot_hrf(fit, voxel, sample_at, level, direct_labels, figsize, ax, text)
    if kind == "timecourse":
        if ax is not None:
            raise ValueError("The time-course view draws several panels; omit `ax`.")
        return _plot_timecourse(fit, voxel, figsize, text)
    return _plot_residual_acf(fit, voxel, figsize, ax, text)


def _header(text: dict[str, str | None], title: str, subtitle: str | None, caption: str | None) -> tuple[str | None, str | None, str | None]:
    pick = [text[k] if text[k] is not None else d for k, d in (("title", title), ("subtitle", subtitle), ("caption", caption))]
    return pick[0], pick[1], pick[2]


def _finish(fig: Figure, ax: Axes | None, text: dict[str, str | None], title: str, subtitle: str | None,
            caption: str | None) -> Figure:
    t, s, c = _header(text, title, subtitle, caption)
    if ax is not None:
        if t:
            ax.set_title(t, loc="left")
        return freeze_fonts(fig)
    return _finish_figure(fig, t, s, c)


def _balanced(text: str | None, width: int) -> str | None:
    """Wrap each paragraph into lines of similar length (R ``.wrap_notes``),
    so no line is left holding an orphaned word."""
    if not text:
        return text
    out = []
    for para in str(text).split("\n"):
        n_lines = math.ceil(len(para) / max(width, 1))
        out.append(para if n_lines <= 1 else textwrap.fill(para, math.ceil(len(para) / n_lines) + 6))
    return "\n".join(out)


def _finish_figure(fig: Figure, title: str | None, subtitle: str | None, caption: str | None) -> Figure:
    """:func:`finish` with balanced captions and fonts pinned."""
    w_in = fig.get_size_inches()[0]
    chars = max(int(w_in * 72 / (BASE_SIZE * 0.52)), 20)
    freeze_fonts(fig)
    return finish(fig, title, _balanced(subtitle, chars), _balanced(caption, int(chars * 1.18)))


def _style_axes(ax: Axes, grid: str = "y") -> None:
    ax.grid(False)
    if grid in ("x", "xy"):
        ax.grid(True, axis="x")
    if grid in ("y", "xy"):
        ax.grid(True, axis="y")
    ax.tick_params(length=0)


# --------------------------------------------------------------------------
# Estimates / contrasts
# --------------------------------------------------------------------------


def _stats_matrices(fit: FmriLm, kind: str) -> dict[str, Any]:
    """Estimates, SEs, t and labels in the voxels x terms layout."""
    if kind == "betas":
        cols = _event_cols(fit)
        if cols.size == 0:
            raise ValueError("This fit has no betas to plot.")
        info = _event_info(fit)
        alias = aliased_columns(fit)
        est = _masked(fit.betas, alias)[cols].T
        se = _masked(fit.se(), alias)[cols].T
        t = _masked(fit.tstat(), alias)[cols].T
        labels = list(info["full_label"])
        groups = [str(g) if g else "" for g in info["term_label"]]
        cmap = condition_colours(info)
        colours = [cmap[lab] for lab in info["label"]]
        df = np.full(t.shape[1], _inference_df(fit))
    else:
        res = [r for r in getattr(fit, "contrasts", {}).values() if getattr(r, "stat_type", None) == "t"]
        if not res:
            raise ValueError("This fit has no t contrasts to plot.")
        est = np.column_stack([np.ravel(r.estimate) for r in res])
        se = np.column_stack([np.ravel(r.se) if r.se is not None else np.full(est.shape[0], np.nan) for r in res])
        t = np.column_stack([np.ravel(r.stat) for r in res])
        labels = [str(r.name) for r in res]
        groups = ["contrasts"] * len(res)
        colours = [CATEGORICAL[0]] * len(res)
        df = np.array([float(r.df) if np.isscalar(r.df) else _inference_df(fit) for r in res])
    seen: dict[str, int] = {}
    uniq = []
    for lab in labels:
        k = seen.get(lab, 0)
        uniq.append(lab if k == 0 else f"{lab} #{k + 1}")
        seen[lab] = k + 1
    return {"est": est, "se": se, "t": t, "labels": uniq, "groups": groups, "colours": colours, "df": df}


def _plot_stats(fit: FmriLm, kind: str, voxel: VoxelArg, level: float, figsize: Size, ax: Axes | None,
                text: dict[str, str | None]) -> Figure:
    m = _stats_matrices(fit, kind)
    what = "Regressor" if kind == "betas" else "Contrast"
    if voxel is None:
        return _plot_t_distribution(fit, m, what, figsize, ax, text)
    return _plot_coefficients(fit, m, what, _check_voxels(fit, voxel), level, figsize, ax, text)


def _plot_t_distribution(fit: FmriLm, m: dict[str, Any], what: str, figsize: Size, ax: Axes | None,
                         text: dict[str, str | None]) -> Figure:
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    t = m["t"]
    nvox, k = t.shape
    labels = m["labels"]
    df_med = float(np.median(m["df"]))
    crit = critical_t(df_med)
    pos, neg, estimable = share_beyond(t, crit)
    show_neg = bool(np.nansum(neg) > 0)
    finite_t = t[np.isfinite(t)]
    lo = min(float(finite_t.min()) if finite_t.size else 0.0, -crit)
    hi = max(float(finite_t.max()) if finite_t.size else 0.0, crit)
    pad = 0.04 * (hi - lo)
    if figsize is None:
        figsize = (7.2, max(2.9, 1.75 + 0.36 * k))
    with themed():
        fig, ax0 = (ax.figure, ax) if ax is not None else new_figure(figsize=figsize)
        ax0.axvline(0.0, color=INK["zero"], linewidth=0.8, zorder=1.5)
        for c in (-crit, crit):
            ax0.axvline(c, color=INK["muted"], linewidth=0.6, dashes=(3, 2.5), zorder=1.5)
        if nvox <= BIN_ABOVE:
            alpha = max(0.15, min(0.8, 12 / math.sqrt(nvox)))
            for j in range(k):
                tj = t[:, j]
                ok = np.isfinite(tj)
                if not ok.any():
                    continue
                y = j + sina_offsets(tj[ok])
                ax0.scatter(tj[ok], y, s=10, alpha=alpha, linewidths=0, color=m["colours"][j], zorder=3)
        else:
            edges = np.linspace(lo - pad, hi + pad, 121)
            counts = np.stack([np.histogram(t[np.isfinite(t[:, j]), j], bins=edges)[0] for j in range(k)])
            top = counts.max() or 1
            patches, faces = [], []
            for j in range(k):
                for b in np.flatnonzero(counts[j]):
                    patches.append(Rectangle((edges[b], j - 0.4), edges[b + 1] - edges[b], 0.8))
                    faces.append(to_rgba(m["colours"][j], 0.05 + 0.95 * counts[j, b] / top))
            ax0.add_collection(PatchCollection(patches, facecolors=faces, edgecolors="none", zorder=3))
        ax0.set_xlim(lo - pad, hi + pad)
        ax0.set_ylim(k - 0.4, -0.6)
        ax0.set_yticks(range(k))
        ax0.set_yticklabels(labels, color=INK["text"], fontsize=BASE_SIZE * 0.85)
        _style_axes(ax0, grid="x")
        ax0.set_xlabel("t-statistic")
        # right margin: share of voxels beyond each line
        cols = [("t > " + _signed(crit, ".2f"), pos)]
        if show_neg:
            cols.append(("t < " + _signed(-crit, ".2f"), neg))
        x_pt = 0.0
        for head, share in cols:
            x_pt += 62.0
            row_t = _margin_transform(ax0, x_pt)
            head_t = _margin_transform(ax0, x_pt, axes_y=True)
            ax0.text(1.0, 1.0, head, transform=head_t, ha="right", va="bottom",
                     fontsize=BASE_SIZE * 0.8, color=INK["text2"], clip_on=False)
            for j in range(k):
                txt = (_pct(share[j]) if estimable[j] else ("aliased" if head.startswith("t >") else ""))
                ax0.text(1.0, j, txt, transform=row_t, ha="right", va="center",
                         fontsize=BASE_SIZE * 0.82, color=INK["text2"], clip_on=False)
        _nice_x(ax0, 8)
    df_txt = _df_text(m["df"])
    notes = [
        f"df = {df_txt}. Under the null, 0.05% of voxels fall beyond each line."
        + ("" if show_neg else f" No voxel has t below {_signed(-crit, '.2f')}."),
        _strategy_text(fit),
    ]
    return _finish(
        fig, ax, text,
        f"{what} t-statistics across {nvox:,} voxels",
        f"Dashed: |t| = {crit:.2f} (p < .001, two-sided). Right: % of voxels beyond.",
        _join_notes(notes),
    )


def _plot_coefficients(fit: FmriLm, m: dict[str, Any], what: str, voxels: list[int], level: float,
                       figsize: Size, ax: Axes | None, text: dict[str, str | None]) -> Figure:
    labels = m["labels"]
    groups = m["groups"]
    group_order = list(dict.fromkeys(groups))
    multi_group = len(group_order) > 1
    nv = len(voxels)
    if ax is not None and (multi_group or nv > 1):
        raise ValueError("This coefficient plot draws several panels; omit `ax`.")
    rows_per = [sum(g == grp for g in groups) for grp in group_order]
    if figsize is None:
        figsize = (min(4.4 * nv + 2.6, 12.0), max(2.6, 1.55 + sum(0.34 * r + 0.55 for r in rows_per)))
    with themed():
        if ax is not None:
            fig, axes = ax.figure, np.array([[ax]])
        else:
            fig, axes = new_figure(len(group_order), nv, figsize=figsize, squeeze=False,
                                   gridspec_kw={"height_ratios": [r + 0.9 for r in rows_per]})
        for gi, grp in enumerate(group_order):
            idx = [j for j, g in enumerate(groups) if g == grp]
            for vi, v in enumerate(voxels):
                a = axes[gi, vi]
                tab = coefficient_intervals(m["est"][v, idx], m["se"][v, idx], m["t"][v, idx],
                                            m["df"][idx], level)
                a.axvline(0.0, color=INK["zero"], linewidth=0.8, zorder=1.5)
                for r, j in enumerate(idx):
                    row = tab.iloc[r]
                    colour = m["colours"][j]
                    if np.isfinite(row["est"]):
                        if np.isfinite(row["lo"]) and np.isfinite(row["hi"]):
                            a.plot([row["lo"], row["hi"]], [r, r], color=tints(colour, 3)[1],
                                   linewidth=2.4, solid_capstyle="butt", zorder=2)
                        a.scatter([row["est"]], [r], s=34, color=colour, zorder=3, linewidths=0)
                    a.text(1.0, r, row["txt"], transform=_margin_transform(a, 8.0), ha="left",
                           va="center", fontsize=BASE_SIZE * 0.8, color=INK["text2"], clip_on=False)
                a.set_ylim(len(idx) - 0.4, -0.6)
                a.set_yticks(range(len(idx)))
                a.set_yticklabels([labels[j] for j in idx] if vi == 0 else [],
                                  color=INK["text"], fontsize=BASE_SIZE * 0.85)
                _style_axes(a, grid="x")
                finite = np.concatenate([tab[c].to_numpy() for c in ("lo", "hi", "est")])
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lo_x, hi_x = min(finite.min(), 0.0), max(finite.max(), 0.0)
                    span = (hi_x - lo_x) or 1.0
                    a.set_xlim(lo_x - 0.06 * span, hi_x + 0.06 * span)
                _nice_x(a, 6)
                name = [grp] if multi_group else []
                if nv > 1:
                    name.append(f"voxel {v}")
                if name:
                    a.set_title(f" {DOT} ".join(name), loc="left", fontsize=BASE_SIZE * 0.85)
                if gi == len(group_order) - 1:
                    a.set_xlabel("Estimate (β)" if what == "Regressor" else "Contrast estimate")
    sub = f"Point: estimate. Line: {100 * level:g}% CI." + (" Each panel has its own units." if multi_group else "")
    return _finish(
        fig, ax, text,
        f"{what} estimates, voxel {voxels[0]}" if nv == 1 else f"{what} estimates",
        sub,
        _join_notes([f"df = {_df_text(m['df'])}.", _strategy_text(fit)]),
    )


def _margin_transform(ax: Axes, points: float, axes_y: bool = False) -> Transform:
    """x at the axes' right edge plus ``points``; y in data (or axes) units."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    base = ax.transAxes if axes_y else blended_transform_factory(ax.transAxes, ax.transData)
    return base + ScaledTranslation(points / 72.0, 0.0, ax.figure.dpi_scale_trans)


def _nice_x(ax: Axes, nbins: int = 6) -> None:
    """Round x ticks (steps of 1, 2 or 5) with one precision per axis."""
    from matplotlib.ticker import MaxNLocator

    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10], min_n_ticks=3))
    ticks = [t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9]
    ax.xaxis.set_major_formatter(axis_number_formatter(ticks))
    ax.set_xlim(lo, hi)


# --------------------------------------------------------------------------
# HRF
# --------------------------------------------------------------------------


def _text_size_data(ax: Axes, txt: str, size: float) -> tuple[float, float]:
    """Width and height of ``txt`` in ``ax`` data units."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    art = ax.text(0, 0, txt, fontsize=size, fontweight="bold", family=theme_rc()["font.sans-serif"])
    bb = art.get_window_extent(renderer)
    art.remove()
    box = ax.get_window_extent(renderer)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return bb.width / box.width * (x1 - x0), bb.height / box.height * (y1 - y0)


def place_curve_labels(
    curves: pd.DataFrame,
    labs: pd.DataFrame,
    widths: Sequence[float],
    height: float,
    xlim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Place one label per curve where no other curve or label passes.

    Port of fmrireg ``.place_curve_labels``: beside the peak dot (right,
    then left), moved away from zero (above a positive peak, below a
    negative one) only as far as needed for the text box to clear the other
    conditions' curves, its own curve and the labels placed already.
    ``widths`` are text widths and ``height`` the clearance, in data units.
    When no spot is clear (many crowded curves) the spot with the fewest
    collisions is used rather than R's fixed fallback. Returns ``x`` (left
    edge) and ``y`` (vertical centre).
    """
    xr = xlim or (float(curves["time"].min()), float(curves["time"].max()))
    vals = curves[["mean", "lo", "hi"]].to_numpy(dtype=float)
    yr = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    dx, dy = xr[1] - xr[0], yr[1] - yr[0]
    out = pd.DataFrame({"x": labs["peak_t"].to_numpy(dtype=float), "y": labs["peak_v"].to_numpy(dtype=float)})
    if dx <= 0 or dy <= 0:
        return out
    h = float(height)
    gap = 0.02 * dx
    boxes: list[tuple[float, float, float]] = []
    order = np.argsort(-np.abs(labs["peak_v"].to_numpy(dtype=float)), kind="stable")
    lifts = (0, 1, -1, 2, -2, 3, -3, 4, 5, 6)
    cands = [(side, lift, push) for push in (0.0, 0.05, 0.1, 0.2, 0.3) for lift in lifts for side in (1, -1)]
    for i in order:
        cond = labs["condition"].iloc[i]
        pt, pv = float(labs["peak_t"].iloc[i]), float(labs["peak_v"].iloc[i])
        sign = 1.0 if pv >= 0 else -1.0
        others = curves[curves["condition"] != cond]
        own = curves[curves["condition"] == cond]
        w = float(widths[i])
        best: tuple[float, float, float] | None = None
        for side, lift, push in cands:
            x0 = pt + gap + push * dx if side > 0 else pt - gap - push * dx - w
            if x0 < xr[0] or x0 + w > xr[1]:
                continue
            yc = pv + sign * (lift + 0.9) * h
            if yc > yr[1] + 3 * h or yc < yr[0] - 3 * h:
                continue
            span = (others["time"] >= x0) & (others["time"] <= x0 + w)
            own_span = (own["time"] >= x0) & (own["time"] <= x0 + w)
            n_curve = int(np.sum(np.abs(others.loc[span, "mean"].to_numpy() - yc) < h)) + int(
                np.sum(np.abs(own.loc[own_span, "mean"].to_numpy() - yc) < 0.7 * h))
            n_label = sum(b0 < x0 + w and x0 < b1 and abs(by - yc) < 2 * h for b0, b1, by in boxes)
            if n_curve == 0 and n_label == 0:
                best = (0.0, x0, yc)
                break
            cost = 1000.0 * n_label + n_curve + abs(lift) + 10 * push
            if best is None or cost < best[0]:
                best = (cost, x0, yc)
        if best is None:
            best = (0.0, min(pt + gap, xr[1] - w), pv + sign * 0.9 * h)
        out.iloc[i] = (best[1], best[2])
        boxes.append((best[1], best[1] + w, best[2]))
    return out


def _plot_hrf(fit: FmriLm, voxel: VoxelArg, sample_at: Sequence[float], level: float, direct_labels: bool,
              figsize: Size, ax: Axes | None, text: dict[str, str | None]) -> Figure:
    import matplotlib.patheffects as pe

    auto = voxel is None
    voxels = [peak_voxel(fit)] if auto else _check_voxels(fit, voxel)
    sample = np.asarray(sample_at, dtype=float)
    curves, lines, aliased, terms = hrf_curves(fit, voxels, sample, level)
    if not terms:
        raise ValueError("This fit has no event terms with an HRF to plot.")
    if curves.empty:
        raise ValueError("No condition in this fit has an estimable response for the selected voxel(s).")
    single = len(voxels) == 1
    info = column_info(fit)
    cmap = condition_colours(info)
    term_order = [td["term"] for td in terms if td["term"] in set(curves["term"])]
    term_label = {td["term"]: td["term_label"] for td in terms}
    n_panels = len(term_order)
    if ax is not None and n_panels > 1:
        raise ValueError("This fit has several event terms, so the HRF view draws several panels; omit `ax`.")
    ncol = 1 if n_panels == 1 else (2 if n_panels <= 4 else 3)
    nrow = math.ceil(n_panels / ncol)
    if figsize is None:
        figsize = (7.2 if ncol == 1 else min(4.0 * ncol + 0.6, 12.0), 1.9 + 2.75 * nrow if nrow > 1 else 4.6)
    size = BASE_SIZE * 0.86
    with themed():
        if ax is not None:
            fig, axes = ax.figure, [ax]
        else:
            fig, grid = new_figure(nrow, ncol, figsize=figsize, squeeze=False)
            axes = list(grid.ravel())
            for extra in axes[n_panels:]:
                extra.set_visible(False)
            axes = axes[:n_panels]
        labs_by_panel = []
        for a, tn in zip(axes, term_order, strict=False):
            cdf = curves[curves["term"] == tn]
            crowded = cdf["condition"].nunique() > 1
            # overlapping fills blend into colours that are in no key, so
            # keep them faint where conditions share a panel
            band_alpha = 0.07 if crowded else 0.15
            a.axhline(0.0, color=INK["zero"], linewidth=0.8, zorder=1.4)
            if not single and not lines.empty:
                tl = lines[lines["term"] == tn]
                n_lines = max(tl.groupby(["condition", "voxel"]).ngroups, 1)
                line_alpha = max(0.04, min(0.12, 2.2 / math.sqrt(n_lines)))
                for (cond, _v), g in tl.groupby(["condition", "voxel"], sort=False):
                    a.plot(g["time"], g["value"], color=cmap.get(cond, INK["text2"]), alpha=line_alpha,
                           linewidth=0.4, zorder=1.6)
            labs = []
            for cond, g in cdf.groupby("condition", sort=False):
                colour = cmap.get(cond, INK["text2"])
                if np.isfinite(g["lo"]).any():
                    a.fill_between(g["time"], g["lo"], g["hi"], color=colour, alpha=band_alpha,
                                   linewidth=0, zorder=2)
                    if crowded and single:
                        for edge in ("lo", "hi"):
                            a.plot(g["time"], g[edge], color=colour, linewidth=0.5, alpha=0.7,
                                   dashes=(2, 2), zorder=2.2)
                sig = bool(g["sig"].iloc[0])
                halo = [] if single else [pe.withStroke(linewidth=3.4, foreground="white")]
                a.plot(g["time"], g["mean"], color=colour, linewidth=1.6, alpha=1.0 if sig else 0.55,
                       **({} if sig else {"dashes": (3, 2)}), zorder=3, label=cond, path_effects=halo)
                pt, pv = float(g["peak_t"].iloc[0]), float(g["peak_v"].iloc[0])
                a.scatter([pt], [pv], s=16, color=colour, zorder=4, linewidths=0)
                nb = int(g["nbasis"].iloc[0])
                txt = f"{cond} (n.s.)" if not sig else (f"{cond}  {pt:.1f} s" if nb > 1 else str(cond))
                labs.append({"condition": cond, "txt": txt, "peak_t": pt, "peak_v": pv, "colour": colour,
                             "sig": sig})
            labs_by_panel.append((a, cdf, pd.DataFrame(labs)))
            vals = cdf[["mean", "lo", "hi"]].to_numpy()
            if not single and not lines.empty:
                vals = np.concatenate([vals.ravel(), lines.loc[lines["term"] == tn, "value"].to_numpy()])
            vals = np.asarray(vals, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            y0, y1 = min(vals.min(), 0.0), max(vals.max(), 0.0)
            span = (y1 - y0) or 1.0
            a.set_ylim(y0 - 0.08 * span, y1 + 0.14 * span)
            a.set_xlim(float(sample.min()), float(sample.max()))
            _nice_x(a, 5)
            _style_axes(a, grid="y")
            budget_yticks(a)
            if n_panels > 1:
                a.set_title(term_label.get(tn, tn), loc="left", fontsize=BASE_SIZE * 0.88)
        for i, a in enumerate(axes):
            if i // ncol == nrow - 1 or i + ncol >= n_panels:
                a.set_xlabel("Time since onset (s)")
            if i % ncol == 0:
                a.set_ylabel("Response (a.u. per unit regressor)" if nrow == 1 else "Response (a.u.)")
        if not direct_labels:
            handles, names = [], []
            seen = set()
            for _a, _cdf, labs in labs_by_panel:
                for _, r in labs.iterrows():
                    if r["condition"] in seen:
                        continue
                    seen.add(r["condition"])
                    from matplotlib.lines import Line2D

                    handles.append(Line2D([], [], color=r["colour"], linewidth=1.6,
                                          alpha=1.0 if r["sig"] else 0.55,
                                          **({} if r["sig"] else {"dashes": (3, 2)})))
                    names.append(r["txt"])
            axes[0].legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.0 if n_panels == 1 else 1.12),
                           ncol=min(len(names), 4), frameon=False, borderaxespad=0.2)
    basis_txt = "/".join(dict.fromkeys(td["hrf_name"] for td in terms))
    multi = bool((curves["nbasis"] > 1).any())
    if single:
        sub = f"Voxel {voxels[0]}{' (largest |t|)' if auto else ''}, {basis_txt} basis. Band: {100 * level:g}% CI."
    else:
        sub = f"Mean of {len(voxels)} voxels, {basis_txt} basis. Band: {100 * level:g}% CI of the mean."
    notes = [
        "Dashed, faded curve: its interval includes 0 at the peak (n.s.)." if single else
        f"Faint lines: {f'{MAX_HRF_LINES} of the voxels' if len(voxels) > MAX_HRF_LINES else 'each voxel'}. "
        "Voxels are treated as independent.",
        "Dots mark each peak; labels give its estimated latency." if multi else
        "One basis function fixes the response shape, so peak latency is not estimated.",
        f"Not estimable (aliased), not drawn: {', '.join(aliased)}." if aliased else None,
    ]
    fig = _finish(fig, ax, text, "Estimated hemodynamic response", sub, _join_notes(notes))
    if direct_labels:
        with themed():
            for a, cdf, labs in labs_by_panel:
                if labs.empty:
                    continue
                widths = []
                hgt = 0.0
                for t in labs["txt"]:
                    w, h = _text_size_data(a, t, size)
                    widths.append(w * 1.04)
                    hgt = max(hgt, h)
                y0, y1 = a.get_ylim()
                clear = max(0.62 * hgt, 0.045 * (y1 - y0))
                pos = place_curve_labels(cdf, labs, widths, clear, xlim=a.get_xlim())
                x_lo, x_hi = a.get_xlim()
                near = 0.035 * (x_hi - x_lo)
                for (_, r), (_, p), w in zip(labs.iterrows(), pos.iterrows(), widths, strict=True):
                    halo_text(a, p["x"], p["y"], r["txt"], colour=r["colour"], size=size,
                              ha="left", va="center")
                    # a label pushed away from its peak gets a leader line
                    x_end = p["x"] if p["x"] >= r["peak_t"] else p["x"] + w
                    far_x = abs(x_end - r["peak_t"]) > near
                    far_y = abs(p["y"] - r["peak_v"]) > 1.6 * clear
                    if far_x or far_y:
                        a.plot([r["peak_t"], x_end], [r["peak_v"], p["y"]], color=r["colour"],
                               linewidth=0.5, alpha=0.8, zorder=4.5)
                top = max(y1, float(pos["y"].max()) + 0.7 * hgt)
                bottom = min(y0, float(pos["y"].min()) - 0.7 * hgt)
                if (bottom, top) != (y0, y1):
                    a.set_ylim(bottom, top)
                    budget_yticks(a)
        freeze_fonts(fig)
    return fig


# --------------------------------------------------------------------------
# Time course
# --------------------------------------------------------------------------


def _dodge(y: Sequence[float], gap: float) -> FloatArray:
    """Push label positions apart to at least ``gap`` (R ``.dodge_labels``)."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y
    o = np.argsort(-y, kind="stable")
    z = y[o].copy()
    for k in range(1, z.size):
        if z[k - 1] - z[k] < gap:
            z[k] = z[k - 1] - gap
    out = y.copy()
    out[o] = z
    return out


def _dodge_within(y: Sequence[float], gap: float, lo: float, hi: float) -> FloatArray:
    """Two-label :func:`_dodge` that keeps both positions within ``[lo, hi]``."""
    y = np.clip(np.asarray(y, dtype=float), lo, hi)
    if y.size != 2 or abs(y[0] - y[1]) >= gap:
        return y
    first_up = y[0] >= y[1]
    top = min(max(y.max(), lo + gap), hi)
    pair = np.array([top, top - gap])
    return pair if first_up else pair[::-1]


def _run_bounds(time: FloatArray, lens: Sequence[int]) -> list[float]:
    dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
    inner = [float(time[s]) - dt / 2 for s in starts[1:]]
    return [float(time[0]), *inner, float(time[-1])]


def _per_run(ax: Axes, time: FloatArray, values: FloatArray, run_id: FloatArray, *, color: str,
             linewidth: float, zorder: float) -> None:
    """One line per run, so no line crosses a run boundary."""
    for r in dict.fromkeys(run_id.tolist()):
        sel = run_id == r
        ax.plot(time[sel], values[sel], color=color, linewidth=linewidth, zorder=zorder)


def _plot_timecourse(fit: FmriLm, voxel: VoxelArg, figsize: Size, text: dict[str, str | None]) -> Figure:
    auto = voxel is None
    v = peak_voxel(fit) if auto else _check_voxels(fit, voxel, single=True)[0]
    model = _model(fit)
    y = _voxel_data(model, [v], "the voxel's data")[:, 0]
    Xe, Xb = _design_parts(model)
    lens = _run_lengths(model)
    run_id = np.repeat(np.arange(1, len(lens) + 1), lens)
    runwise = _is_runwise(fit)
    rf = voxel_refit(y, Xe, Xb, run_id, runwise)
    time = _scan_times(model, y.size)
    info = column_info(fit)
    ev = _event_onsets(model, info)
    has_lanes = ev is not None and len(ev) > 0
    sig_vals = np.concatenate([y, rf["fitted"]])
    s_lo, s_hi = float(np.nanmin(sig_vals)), float(np.nanmax(sig_vals))
    s_rng = (s_hi - s_lo) or 1.0
    r_lo, r_hi = float(np.nanmin(rf["resid"])), float(np.nanmax(rf["resid"]))
    r_rng = (r_hi - r_lo) or s_rng * 0.1
    lv = list(ev["condition"].cat.categories) if has_lanes else []
    lane_rng = 0.1 * s_rng * (len(lv) + 0.4)
    ratios = [s_rng, lane_rng, r_rng] if has_lanes else [s_rng, r_rng]
    total = sum(ratios)
    floors = [0.2, 0.05 * (len(lv) + 0.4), 0.14] if has_lanes else [0.2, 0.14]
    ratios = [max(r / total, f) for r, f in zip(ratios, floors, strict=True)]
    if figsize is None:
        figsize = (8.2, 5.0 + (0.12 * max(len(lv) - 3, 0)))
    cmap = condition_colours(info)
    bounds = _run_bounds(time, lens)
    dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    x_lab = float(time[-1]) + dt
    with themed():
        fig, axes = new_figure(len(ratios), 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": ratios})
        a_sig = axes[0]
        a_res = axes[-1]
        _per_run(a_sig, time, y, run_id, color=INK["muted"], linewidth=0.6, zorder=2)
        _per_run(a_sig, time, rf["fitted"], run_id, color=INK["text"], linewidth=1.1, zorder=3)
        a_sig.set_ylim(s_lo - 0.04 * s_rng, s_hi + 0.04 * s_rng)
        _style_axes(a_sig, grid="y")
        budget_yticks(a_sig)
        row_label(a_sig, "Signal")
        n = y.size
        end_y = _dodge([y[n - 1], rf["fitted"][n - 1]], 0.07 * s_rng)
        end_y = _dodge_within(end_y, 0.07 * s_rng, s_lo + 0.02 * s_rng, s_hi - 0.02 * s_rng)
        for yy, txt, col in zip(end_y, ("observed", "fitted"), (INK["text2"], INK["text"]), strict=True):
            a_sig.text(x_lab, yy, txt, ha="left", va="center", color=col, fontsize=BASE_SIZE * 0.8,
                       clip_on=False)
        if has_lanes:
            a_ev = axes[1]
            nl = len(lv)
            for i, cond in enumerate(lv):
                yc = nl - i
                col = cmap.get(cond, INK["text2"])
                ons = ev.loc[ev["condition"] == cond, "onset"].to_numpy()
                a_ev.vlines(ons, yc - 0.38, yc + 0.38, color=col, linewidth=0.9, zorder=3)
                a_ev.text(x_lab, yc, str(cond), ha="left", va="center", color=col,
                          fontsize=BASE_SIZE * 0.8, clip_on=False)
            a_ev.set_ylim(0.3, nl + 0.7)
            a_ev.set_yticks([])
            _style_axes(a_ev, grid="none")
            row_label(a_ev, "Onsets")
        a_res.axhline(0.0, color=INK["zero"], linewidth=0.8, zorder=1.4)
        _per_run(a_res, time, rf["resid"], run_id, color=INK["muted"], linewidth=0.6, zorder=2)
        a_res.set_ylim(r_lo - 0.04 * r_rng, r_hi + 0.04 * r_rng)
        _style_axes(a_res, grid="y")
        budget_yticks(a_res)
        row_label(a_res, "Residual")
        for i, a in enumerate(axes):
            run_axis(a, bounds, top=(i == 0))
            a.tick_params(axis="x", labelbottom=(a is a_res))
        time_axis(a_res, float(time[0]), float(time[-1]))
    r2 = rf["partial_r2"]
    r2_txt = "n/a" if not np.isfinite(r2) else f"{r2:.2f}"
    cap = (f"Fitted: {'per-run' if runwise else 'joint'} OLS of the full design"
           + (" (run-specific coefficients; the estimates view pools runs)" if runwise else "")
           + ". Signal and residual share one scale, in data units.")
    return _finish_figure(
        fig,
        *_header(text, f"Observed and fitted signal, voxel {v}{' (largest |t|)' if auto else ''}",
                 f"Partial R² of events given baseline = {r2_txt}", cap),
    )


# --------------------------------------------------------------------------
# Residual autocorrelation
# --------------------------------------------------------------------------


def _plot_residual_acf(fit: FmriLm, voxel: VoxelArg, figsize: Size, ax: Axes | None, text: dict[str, str | None],
                       max_lag: int = 10) -> Figure:
    from matplotlib.patches import Rectangle

    nvox = int(np.asarray(fit.betas).shape[1])
    user = voxel is not None
    voxels = _check_voxels(fit, voxel) if user else list(
        dict.fromkeys(np.round(np.linspace(0, nvox - 1, min(nvox, MAX_ACF_VOXELS))).astype(int).tolist()))
    model = _model(fit)
    lens = _run_lengths(model)
    lag_max = min(max_lag, min(lens) - 3)
    if lag_max < 1:
        raise ValueError("Runs are too short to estimate residual autocorrelation.")
    Y = _voxel_data(model, voxels, "residuals")
    X = np.asarray(model.design_matrix_array(), dtype=float)
    run_id = np.repeat(np.arange(1, len(lens) + 1), lens)
    runwise = _is_runwise(fit)
    phi, pooled_vox = _fit_ar_phi(fit, len(lens))
    series = {"OLS": residual_acf(Y, X, run_id, runwise, lag_max)}
    if phi is not None:
        series["prewhitened"] = residual_acf(Y, X, run_id, runwise, lag_max, phi=phi)
    single = len(voxels) == 1
    ns = len(series)
    dodge = (-0.17, 0.17) if ns > 1 else (0.0,)
    box_w = 0.15 if ns > 1 else 0.34
    if ns > 1:
        inks = {"OLS": INK["muted"], "prewhitened": INK["text"]}
        filled = {"OLS": False, "prewhitened": True}
    else:
        inks, filled = {"OLS": INK["text"]}, {"OLS": True}
    verdict = []
    extent: list[Any] = []
    if figsize is None:
        figsize = (7.2, 4.4)
    with themed():
        fig, a = (ax.figure, ax) if ax is not None else new_figure(figsize=figsize)
        lags = np.arange(1, lag_max + 1)
        for i, (name, sr) in enumerate(series.items()):
            x = lags + dodge[i]
            lo, hi = acf_band(sr["expect"], sr["cov"])
            extent += [lo, hi]
            for xx, l0, h0, c in zip(x, lo, hi, sr["expect"], strict=True):
                a.add_patch(Rectangle((xx - box_w, l0), 2 * box_w, h0 - l0, facecolor=INK["grid"],
                                      edgecolor="none", zorder=1.2))
                a.plot([xx - box_w, xx + box_w], [c, c], color=INK["muted"], linewidth=0.8,
                       solid_capstyle="butt", zorder=1.6)
            A = sr["acf"]
            if single:
                r = A[:, 0]
            else:
                r = np.nanmedian(A, axis=1)
                q10, q90 = np.nanquantile(A, 0.1, axis=1), np.nanquantile(A, 0.9, axis=1)
                a.vlines(x, q10, q90, color=inks[name], linewidth=1.2, zorder=3)
                extent += [q10, q90]
            extent.append(r)
            a.scatter(x, r, s=30, zorder=4, linewidths=1.0, edgecolors=inks[name],
                      facecolors=inks[name] if filled[name] else "white", label=name)
            q, pv = portmanteau(A, sr["expect"], sr["cov"])
            if single:
                p0 = float(pv[0])
                verdict.append(f"{name} " + ("p < .001" if p0 < 0.001 else f"p = {num(p0, 2)}"))
            else:
                verdict.append(f"{name} {int(np.sum(pv < 0.05))}/{len(voxels)}")
        a.axhline(0.0, color=INK["zero"], linewidth=0.8, zorder=1.4)
        a.set_xlim(0.5, lag_max + 0.5)
        a.set_xticks(lags)
        _style_axes(a, grid="y")
        span_v = np.concatenate([np.ravel(v) for v in extent] + [[0.0]])
        span_v = span_v[np.isfinite(span_v)]
        y0, y1 = float(span_v.min()), float(span_v.max())
        pad = 0.06 * ((y1 - y0) or 1.0)
        a.set_ylim(y0 - pad, y1 + pad)
        budget_yticks(a)
        a.set_xlabel("Lag (scans)")
        a.set_ylabel("Autocorrelation")
        if ns > 1:
            a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                     handletextpad=0.3, borderaxespad=0.2)
    if single:
        title = f"Residual autocorrelation, voxel {voxels[0]}"
        sub = f"Portmanteau test, lags 1–{lag_max}: " + ", ".join(verdict)
    else:
        title = f"Residual autocorrelation, {len(voxels)} {'selected ' if user else ''}voxels"
        sub = (f"Voxels failing the portmanteau test (p < .05): {', '.join(verdict)} "
               f"({num(0.05 * len(voxels), 2)} expected by chance)")
    notes = [
        "Grey: 95% white-noise range for each series under this design, centred on the dash; "
        "the test uses the exact null covariance of the lags."
        + ("" if single else " Points: median across voxels; lines: 10th–90th percentile."),
        None if phi is None else
        ("Prewhitened with the model's AR coefficients ("
         + ("median across voxels" if pooled_vox else "pooled across voxels")
         + "; lag 1 by run: " + ", ".join(_signed(float(p[0]), ".2f") for p in phi) + ")."),
        "ARMA noise model: the prewhitened series is not shown." if getattr(fit, "ma_params", None) is not None
        else None,
    ]
    return _finish(fig, ax, text, title, sub, _join_notes(notes))


# --------------------------------------------------------------------------
# Model regressors over time
# --------------------------------------------------------------------------


def row_breaks(lo: float, hi: float) -> FloatArray:
    """0 plus one rounded value per sign the data reach (R ``.row_breaks``)."""
    m = max(abs(lo), abs(hi))
    if not np.isfinite(m) or m == 0:
        return np.array([0.0])
    # one significant digit, rounded down so the break stays inside the data
    mag = 10.0 ** math.floor(math.log10(0.8 * m))
    b = math.floor(0.8 * m / mag) * mag
    out = [v for v, keep in ((-b, lo < -0.25 * b), (0.0, True), (b, hi > 0.25 * b)) if keep]
    return np.array([v for v in out if lo <= v <= hi])


def plot_fmri_model(
    model: FmriModel | FmriLm,
    baseline: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot a model's regressors over time (R fmrireg ``plot.fmri_model``).

    Stacked small multiples sharing one time axis: one row per event
    column coloured by condition (one row per event term when there are
    more than 12 event columns), and one grey row per baseline term (drift,
    nuisance). Run-specific columns are drawn only inside their run, no
    line crosses a run boundary, and run intercepts are omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.ticker import FixedLocator

    model = _model(model)
    DM = np.asarray(model.design_matrix_array(), dtype=float)
    info = column_info(model)
    n = DM.shape[0]
    lens = _run_lengths(model)
    if sum(lens) != n:
        lens = [n]
    run_id = np.repeat(np.arange(1, len(lens) + 1), lens)
    time = _scan_times(model, n)
    is_event = (info["kind"] == "event").to_numpy()
    is_icpt = (info["role"].astype(str) == "intercept").to_numpy()
    keep = is_event | (baseline & ~is_event & ~is_icpt)
    n_ev = int(is_event.sum())
    per_term = n_ev > 12
    cmap = condition_colours(info)
    panel, colour = [], []
    for j, r in info.iterrows():
        if is_event[j]:
            panel.append(str(r["term_label"]) if per_term else str(r["full_label"]))
            colour.append(cmap[r["label"]])
        else:
            panel.append(str(r["term_label"] or r["term"]))
            colour.append(INK["muted"])
    if per_term:
        crowded = info[is_event].groupby("term_label")["label"].nunique()
        for j in np.flatnonzero(is_event):
            if crowded.get(info.at[j, "term_label"], 0) > 8:
                colour[j] = CATEGORICAL[0]
    # a column is run-specific when it is zero outside one run
    col_run = np.full(DM.shape[1], 0)
    if len(lens) > 1:
        nz = DM != 0
        for j in range(DM.shape[1]):
            runs = np.unique(run_id[nz[:, j]])
            if runs.size == 1:
                col_run[j] = runs[0]
    sel = np.flatnonzero(keep)
    panels = list(dict.fromkeys(panel[j] for j in sel))
    if not panels:
        raise ValueError("The model has no regressors to plot.")
    nrow = len(panels)
    if figsize is None:
        figsize = (8.2, max(2.8, 1.45 + 0.52 * nrow))
    bounds = _run_bounds(time, lens)
    with themed():
        fig, axes = new_figure(nrow, 1, figsize=figsize, sharex=True, squeeze=False,
                               )
        axes = axes[:, 0]
        for a, p in zip(axes, panels, strict=True):
            cols = [j for j in sel if panel[j] == p]
            lo, hi = np.inf, -np.inf
            for j in cols:
                vals = DM[:, j].copy()
                if col_run[j]:
                    vals[run_id != col_run[j]] = np.nan
                for r in range(1, len(lens) + 1):
                    m = (run_id == r) & np.isfinite(vals)
                    if m.any():
                        a.plot(time[m], vals[m], color=colour[j], linewidth=0.8, zorder=3)
                if np.isfinite(vals).any():
                    lo, hi = min(lo, np.nanmin(vals)), max(hi, np.nanmax(vals))
            if not np.isfinite(lo):
                lo, hi = 0.0, 1.0
            span = (hi - lo) or max(abs(hi), 1.0)
            a.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
            _style_axes(a, grid="none")
            ticks = row_breaks(*a.get_ylim())
            a.yaxis.set_major_locator(FixedLocator(ticks))
            a.yaxis.set_major_formatter(axis_number_formatter(ticks))
            a.yaxis.tick_right()
            a.tick_params(axis="y", labelsize=BASE_SIZE * 0.72)
            a.axhline(0.0, color=INK["grid"], linewidth=0.6, zorder=1)
            row_label(a, p, colour=INK["text"])
        for i, a in enumerate(axes):
            run_axis(a, bounds, top=(i == 0))
            a.tick_params(axis="x", labelbottom=(i == nrow - 1))
        time_axis(axes[-1], float(time[0]), float(time[-1]))
    n_bl = int((keep & ~is_event).sum())
    omitted = int((~is_event & is_icpt).sum())
    sub = f"{n_ev} event column{'s' if n_ev != 1 else ''}"
    if baseline:
        sub += f", {n_bl} baseline column{'s' if n_bl != 1 else ''} (grey)"
        if omitted:
            sub += f"; {omitted} run intercept{'s' if omitted != 1 else ''} not shown"
    tr = _tr_text(model)
    cap = (f"Regressors drawn as sampled at each scan (TR = {tr} s); each row has its own vertical scale."
           if tr else "Each row has its own vertical scale.")
    fig = _finish_figure(fig, *_header({"title": title, "subtitle": subtitle, "caption": caption},
                                       "Model regressors over time", sub, cap))
    _open_rows(fig, 0.42)
    return fig


def _open_rows(fig: Figure, hspace: float) -> None:
    """Widen the gap between stacked rows after the tight layout, so the
    tick labels of neighbouring rows never touch."""
    if fig.subplotpars.hspace < hspace:
        fig.subplots_adjust(hspace=hspace)


__all__ = [
    "acf_band",
    "acf_cols",
    "acf_null_cov",
    "ar_whiten",
    "coefficient_intervals",
    "condition_colours",
    "critical_t",
    "hrf_curves",
    "hrf_terms",
    "peak_voxel",
    "place_curve_labels",
    "plot_fit",
    "plot_fmri_model",
    "portmanteau",
    "residual_acf",
    "row_breaks",
    "share_beyond",
    "sina_offsets",
    "voxel_refit",
]
