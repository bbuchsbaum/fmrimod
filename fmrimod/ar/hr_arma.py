"""Hannan-Rissanen ARMA(p,q) estimation.

Ports ``hr_arma.R`` and ``fmriAR_hr.cpp`` into pure NumPy/SciPy.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from .acvf import PooledAcvf, acvf_from_pooled, pooled_acvf_segments, yw_from_acvf
from .numhelpers import (
    enforce_invertible_ma,
    enforce_stationary_ar,
    levinson_durbin,
)
from .whitening import arma_whiten_segments


def _lag_matrix(x: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Build a matrix of lagged values.

    Parameters
    ----------
    x : NDArray
        1-D input, shape ``(n,)``.
    k : int
        Number of lags.

    Returns
    -------
    NDArray
        Matrix shape ``(n, k)`` where column *j* is ``x`` shifted by
        ``j+1`` steps.  Leading rows are filled with 0.
    """
    n = len(x)
    if k <= 0:
        return np.empty((n, 0), dtype=np.float64)
    M = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        lag = j + 1
        M[lag:, j] = x[: n - lag]
    return M


def _arma_innovations(
    y: NDArray[np.float64], phi: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute ARMA innovations (residuals) via filtering.

    Applies the ARMA filter::

        e[t] = y[t] - phi_1*y[t-1] - ... - phi_p*y[t-p]
                     - theta_1*e[t-1] - ... - theta_q*e[t-q]

    Using ``scipy.signal.lfilter``.

    Parameters
    ----------
    y : NDArray
        Input series, shape ``(n,)``.
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    theta : NDArray
        MA coefficients, shape ``(q,)``.

    Returns
    -------
    NDArray
        Innovation series, shape ``(n,)``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()

    # AR polynomial: b = [1, -phi_1, -phi_2, ...]
    b = np.concatenate([[1.0], -phi])
    # MA polynomial: a = [1, theta_1, theta_2, ...]
    a = np.concatenate([[1.0], theta])

    return cast("NDArray[np.float64]", lfilter(b, a, y))


def hr_arma(
    y: NDArray[np.float64],
    p: int,
    q: int,
    *,
    p_big: Optional[int] = None,
    step1: str = "yw",
    n_iter: int = 0,
    bound: float = 0.99,
    enforce: bool = True,
) -> dict[str, Any]:
    """Hannan-Rissanen two-step ARMA estimation.

    Parameters
    ----------
    y : NDArray
        Input series, shape ``(n,)``.
    p : int
        AR order.
    q : int
        MA order.
    p_big : int, optional
        High-order AR for preliminary residuals (default: auto).
    step1 : str
        ``"yw"`` (Yule-Walker) or ``"burg"`` for step-1 AR fit.
    n_iter : int
        Number of refinement iterations after the initial fit.
    bound : float
        PACF clipping bound for stationarity enforcement.
    enforce : bool
        Whether to enforce stationarity/invertibility.

    Returns
    -------
    dict
        ``{"phi", "theta", "sigma2", "order", "method", "p_big", "iterations"}``
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y - y.mean()
    n = len(y)
    if n < 10:
        raise ValueError("Series too short for HR estimation")

    if p_big is None:
        p_big = max(8, p + q + 5, int(np.ceil(10 * np.log10(n))))
        p_big = min(p_big, max(2, n - 2), 40)

    # Step 1: high-order AR to get preliminary residuals
    # Compute autocovariance
    gamma = np.zeros(p_big + 1)
    for lag in range(p_big + 1):
        if lag >= n:
            break
        gamma[lag] = np.sum(y[: n - lag] * y[lag:]) / n

    if gamma[0] < 1e-15:
        phi_init = np.zeros(p_big, dtype=np.float64)
    else:
        phi_init, _ = levinson_durbin(gamma[: p_big + 1], p_big)

    # Compute initial residuals from high-order AR
    ehat = _arma_innovations(y, phi_init, np.array([]))

    # Initialize
    phi = np.zeros(p, dtype=np.float64) if p > 0 else np.array([], dtype=np.float64)
    theta = np.zeros(q, dtype=np.float64) if q > 0 else np.array([], dtype=np.float64)
    m = max(p, q)
    idx = slice(m, n)
    z_y = y[idx]
    Ylags = _lag_matrix(y, p)
    Yblock = Ylags[idx]

    # Step 2 + refinement iterations
    for _iteration in range(n_iter + 1):
        if q > 0:
            Elags = _lag_matrix(ehat, q)
            Z = np.hstack([Yblock, Elags[idx]])
        else:
            Z = Yblock

        if Z.shape[0] < Z.shape[1] + 1:
            raise ValueError(
                f"Not enough data for ARMA({p},{q}): "
                f"n_eff={Z.shape[0]}, n_params={Z.shape[1]}"
            )

        coef, _, _, _ = np.linalg.lstsq(Z, z_y, rcond=None)

        if p > 0:
            phi = coef[:p]
        if q > 0:
            theta = coef[p : p + q]

        # Recompute innovations for next iteration
        ehat = _arma_innovations(y, phi, theta)

    # Enforce stationarity/invertibility
    if enforce:
        if len(phi):
            phi = enforce_stationary_ar(phi, bound=bound)
        if len(theta):
            theta = enforce_invertible_ma(theta)

    sigma2 = float(np.mean(ehat ** 2))

    return {
        "phi": phi,
        "theta": theta,
        "sigma2": sigma2,
        "order": (int(p), int(q)),
        "method": "hr",
        "p_big": int(p_big),
        "iterations": int(n_iter),
    }


# ---------------------------------------------------------------------------
# Pooled, segment-aware Hannan-Rissanen (fmriAR 617bd87, R/hr_pooled.R)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PooledArmaFit:
    """Result of :func:`_hr_arma_pooled`."""

    phi: NDArray[np.float64]
    theta: NDArray[np.float64]
    sigma2: float
    order: tuple[int, int]
    ok: bool


@dataclass(frozen=True)
class _ArmaUnit:
    """Centred valid-frame residuals of one unit and their segmentation."""

    mat: NDArray[np.float64]
    starts0: NDArray[np.intp]
    rel: NDArray[np.intp]
    seg_id: NDArray[np.intp]
    seg_len: NDArray[np.intp]


def _arma_unit(mat: NDArray[Any], starts0: NDArray[Any]) -> _ArmaUnit:
    """Centre a unit's columns (one mean per run, as on the AR path)."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    if mat.shape[0]:
        mat = mat - mat.mean(axis=0)
    starts = np.asarray(starts0, dtype=np.intp).ravel()
    if starts.size == 0:
        starts = np.array([0], dtype=np.intp)
    seg_len = np.diff(np.append(starts, mat.shape[0]))
    rel = np.concatenate([np.arange(k, dtype=np.intp) for k in seg_len])
    seg_id = np.repeat(np.arange(1, seg_len.size + 1, dtype=np.intp), seg_len)
    return _ArmaUnit(mat=mat, starts0=starts, rel=rel, seg_id=seg_id, seg_len=seg_len)


def _effective_voxels(units: Sequence[_ArmaUnit]) -> float:
    """Kish effective number of independent voxels across centred units.

    With mean inter-voxel correlation ``rbar``, ``V_eff = V / (1 + (V - 1)
    rbar)``; ``rbar`` follows from the variance of the voxel mean, so no
    ``V x V`` matrix is formed. Bounded to ``[1, V]``.
    """
    V = units[0].mat.shape[1]
    if V <= 1:
        return 1.0
    vbar = sum(float(np.sum(u.mat * u.mat)) for u in units) / V
    vmean = sum(float(np.sum(u.mat.mean(axis=1) ** 2)) for u in units)
    if not vbar > 0:
        return 1.0
    rbar = min(max((V * vmean / vbar - 1.0) / (V - 1.0), 0.0), 1.0)
    return float(max(1.0, min(float(V), V / (1.0 + (V - 1.0) * rbar))))


def _hr_normal_eq(
    unit: _ArmaUnit, E: NDArray[np.float64], p: int, q: int, start: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    """Hannan-Rissanen normal equations summed over voxels.

    Rows whose position in their segment is at least ``start`` regress
    ``y_t`` on ``y_{t-1..t-p}`` and ``E_{t-1..t-q}``, so no lag crosses a
    segment start (fmriAR ``hr_normal_eq_cpp``). Returns ``(G, b, y'y, rows)``.
    """
    rows = np.flatnonzero(unit.rel >= start)
    Yt = unit.mat[rows]
    Z = [unit.mat[rows - a] for a in range(1, p + 1)]
    Z += [E[rows - a] for a in range(1, q + 1)]
    k = p + q
    G = np.empty((k, k))
    b = np.empty(k)
    for i in range(k):
        b[i] = np.vdot(Z[i], Yt)
        for j in range(i, k):
            G[i, j] = G[j, i] = np.vdot(Z[i], Z[j])
    return G, b, float(np.vdot(Yt, Yt)), int(rows.size)


def _hr_arma_pooled(
    units: Sequence[tuple[NDArray[Any], NDArray[Any]]],
    p_orders: int | Sequence[int],
    q_orders: int | Sequence[int],
    n_iter: int = 0,
    p_big: int | None = None,
    ar_bound: float = 0.99,
) -> _PooledArmaFit:
    """Hannan-Rissanen ARMA pooled over voxels, inside contiguous segments.

    Ports fmriAR 617bd87 / a75a5de ``.hr_arma_pooled``. Every voxel is a
    replicate series sharing ``(phi, theta)``: the long AR comes from the
    pooled, segment-respecting autocovariance and both regressions are
    solved on cross-products summed over voxels. The voxel-mean series it
    replaces is dominated by whatever the voxels share (phi 0.89 for voxels
    whose own lag-1 correlation is 0.30). No lag spans a censoring gap or
    run boundary, and the long-AR residuals are dropped for the first
    ``p_big`` frames of each segment, where the zero pre-sample makes them
    poor innovation proxies.

    ``units`` are ``(valid-frame residuals (rows x voxels), 0-based segment
    starts)`` pairs; columns are centred per unit. ``p_orders`` and
    ``q_orders`` are an AR and an MA order, or candidate grids chosen by BIC
    on the full-lag regression with N = rows x effective voxels. ``sigma2``
    is the voxel-scale mean squared exact-whitened residual. A fit that
    cannot be made warns and returns white noise with ``ok=False``.
    """
    p_grid = sorted({int(v) for v in np.atleast_1d(p_orders)})
    q_grid = sorted({int(v) for v in np.atleast_1d(q_orders)})
    if min(p_grid) < 0 or min(q_grid) < 0:
        raise ValueError("ARMA orders must be non-negative")
    p, q = p_grid[-1], q_grid[-1]
    selecting = len(p_grid) > 1 or len(q_grid) > 1
    prepared = [_arma_unit(m, s) for m, s in units]
    prepared = [u for u in prepared if u.mat.shape[0] >= 2 and u.mat.shape[1] >= 1]

    def failure(reason: str, p_fail: int, q_fail: int) -> _PooledArmaFit:
        warnings.warn(
            f"fit_noise: ARMA estimation failed ({reason}); "
            "returning a white-noise model",
            UserWarning,
            stacklevel=3,
        )
        return _PooledArmaFit(
            phi=np.zeros(p_fail),
            theta=np.zeros(q_fail),
            sigma2=float("nan"),
            order=(p_fail, q_fail),
            ok=False,
        )

    def white() -> _PooledArmaFit:
        ss = sum(float(np.sum(u.mat * u.mat)) for u in prepared)
        cnt = sum(u.mat.size for u in prepared)
        empty = np.array([], dtype=np.float64)
        return _PooledArmaFit(
            phi=empty, theta=empty, sigma2=ss / cnt, order=(0, 0), ok=True
        )

    if not prepared:
        return failure("no valid data", p, q)
    k = p + q
    if k == 0:
        return white()

    seg_lens = np.concatenate([u.seg_len for u in prepared])

    def usable_rows(pb: int) -> int:
        return int(np.sum(np.maximum(0, seg_lens - max(p, pb + q))))

    if p_big is None:
        p_big = min(max(8, k + 5, int(np.ceil(10 * np.log10(seg_lens.max())))), 40)
    p_big = int(min(p_big, seg_lens.max() - 1))
    # Shrink the long-AR order until enough rows survive the burn-in.
    while p_big > max(1, k) and usable_rows(p_big) < 10 * (k + 1):
        p_big -= 1
    if usable_rows(p_big) <= k:
        return failure("too few usable timepoints", p, q)

    # Step 1: long AR from the pooled, segment-respecting autocovariance.
    num = np.zeros(p_big + 1)
    pairs = np.zeros(p_big + 1)
    for u in prepared:
        pl = pooled_acvf_segments(u.mat, u.seg_id, p_big)
        num += pl.num * u.mat.shape[1]
        pairs += pl.pairs * u.mat.shape[1]
    g, _ = acvf_from_pooled(PooledAcvf(num=num, pairs=pairs), order=p_big)
    if g.size < 2:
        return failure("degenerate autocovariance", p, q)
    pb = min(p_big, g.size - 1)
    phi_big, _ = yw_from_acvf(g[: pb + 1], pb)
    no_ma = np.array([], dtype=np.float64)
    E_list = [arma_whiten_segments(u.mat, phi_big, no_ma, u.starts0) for u in prepared]

    def normal_eq(pp: int, qq: int, start: int) -> tuple[Any, Any, float, int]:
        G = np.zeros((pp + qq, pp + qq))
        b = np.zeros(pp + qq)
        yy, rows = 0.0, 0
        for u, E in zip(prepared, E_list, strict=True):
            Gu, bu, yyu, ru = _hr_normal_eq(u, E, pp, qq, start)
            G += Gu
            b += bu
            yy += yyu
            rows += ru
        return G, b, yy, rows

    if selecting:
        # One regression on the full lag set over common rows; every (p, q)
        # candidate is a column subset: RSS = y'y - b_s' coef_s.
        G, b, yy, rows = normal_eq(p, q, max(p, pb + q))
        V = prepared[0].mat.shape[1]
        if rows < 2:
            return failure("too few usable timepoints", p, q)
        # Voxels are replicates but not independent ones.
        n_bic = rows * _effective_voxels(prepared)
        # R which.min(): the first candidate (p fastest) unless one is finite.
        best: tuple[float, int, int] = (np.inf, p_grid[0], q_grid[0])
        for qq in q_grid:
            for pp in p_grid:
                cols = list(range(pp)) + [p + j for j in range(qq)]
                rss = yy
                if cols:
                    try:
                        cf = np.linalg.solve(G[np.ix_(cols, cols)], b[cols])
                    except np.linalg.LinAlgError:
                        continue
                    if not np.all(np.isfinite(cf)):
                        continue
                    rss = yy - float(b[cols] @ cf)
                s2 = rss / (rows * V)
                if not np.isfinite(s2) or s2 <= 0:
                    continue
                bic = n_bic * np.log(s2) + (pp + qq) * np.log(n_bic)
                if bic < best[0]:
                    best = (bic, pp, qq)
        p, q = best[1], best[2]
        if p + q == 0:
            return white()

    phi = np.zeros(p)
    theta = np.zeros(q)
    for it in range(int(n_iter) + 1):
        G, b, _, _ = normal_eq(p, q, max(p, pb + q))
        try:
            coef = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return failure("singular regression", p, q)
        if not np.all(np.isfinite(coef)):
            return failure("singular regression", p, q)
        phi = coef[:p]
        theta = coef[p:]
        if p:
            phi = enforce_stationary_ar(phi, ar_bound)
            if phi.size != p:
                phi = np.zeros(p)
        if q:
            theta = enforce_invertible_ma(theta)
        if it < n_iter:
            E_list = [
                arma_whiten_segments(u.mat, phi, theta, u.starts0) for u in prepared
            ]

    # Innovation variance at voxel scale, from the exact (stationary-start)
    # whitening of every valid frame.
    ss, cnt = 0.0, 0
    for u in prepared:
        W = arma_whiten_segments(u.mat, phi, theta, u.starts0, exact_first_ar1=True)
        ss += float(np.sum(W * W))
        cnt += W.size
    return _PooledArmaFit(
        phi=np.asarray(phi, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        sigma2=ss / cnt,
        order=(p, q),
        ok=True,
    )
