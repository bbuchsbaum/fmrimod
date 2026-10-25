"""Shared helpers for the fmriAR drift-v2 verification scripts.

R transcriptions of fmriAR 0.4.2 (bb0789c) formulas, built on fmrimod's own
wave1 helpers (pooled_acvf_segments, acvf_from_pooled, yw_from_acvf,
enforce_stationary_ar) wherever R reuses the same 0.3.3 helper.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from fmrimod.ar.acvf import (
    acvf_from_pooled,
    acvf_max_lag,
    pooled_acvf_segments,
    yw_from_acvf,
)
from fmrimod.ar.numhelpers import enforce_stationary_ar


def sim_arma(n, v, phi=(), theta=(), rng=None, burn=200):
    """arima.sim analogue: y = ARMA(phi, theta) with unit innovations."""
    rng = np.random.default_rng() if rng is None else rng
    e = rng.standard_normal((n + burn, v))
    b = np.concatenate([[1.0], np.asarray(theta, float)])
    a = np.concatenate([[1.0], -np.asarray(phi, float)])
    return lfilter(b, a, e, axis=0)[burn:]


# --- R .bic_n (fit_and_apply.R, 61b89ff) ------------------------------------
def r_bic_n(mat, n_frames, groups=None, vbar=None):
    mat = np.asarray(mat, float)
    if mat.ndim == 1:
        mat = mat[:, None]
    V = mat.shape[1]
    if V <= 1 or mat.shape[0] < 2:
        return float(n_frames)
    grp = np.ones(mat.shape[0], int) if groups is None else np.asarray(groups)
    rm = mat.mean(axis=1)
    vmean = 0.0
    for g in np.unique(grp):
        r = rm[grp == g]
        vmean += np.sum((r - r.mean()) ** 2)
    if vbar is None:
        vbar = 0.0
        for g in np.unique(grp):
            m = mat[grp == g]
            vbar += np.sum((m * m).sum(0) - m.shape[0] * m.mean(0) ** 2) / V
    if not vbar > 0:
        return float(n_frames)
    rbar = min(max((V * vmean / vbar - 1) / (V - 1), 0.0), 1.0)
    return float(n_frames * max(1.0, min(V, V / (1 + (V - 1) * rbar))))


def r_select_ar_run(mat, p_max=6, p="auto", correction=None):
    """R 0.4.2 est_run (method='ar', no censoring) for one run's matrix."""
    mat = np.asarray(mat, float)
    if mat.ndim == 1:
        mat = mat[:, None]
    n = mat.shape[0]
    if p != "auto":
        p_max = max(int(p_max), int(p))
    p_cap = min(int(p_max), n - 1)
    seg_id = np.ones(n, int)
    lag_budget = p_cap if correction is None else min(max(p_cap, correction.shape[0] - 1), n - 1)
    pooled = pooled_acvf_segments(mat, seg_id, lag_budget)
    gamma0, _ = acvf_from_pooled(pooled, order=0, correction=correction)
    p_cap = min(p_cap, acvf_max_lag(pooled))

    def fit_order(pp):
        g, _ = acvf_from_pooled(pooled, order=pp, correction=correction)
        phi, s2 = yw_from_acvf(g[: pp + 1], pp)
        return phi, max(s2, 1e-12)

    if p != "auto":
        pp = min(int(p), p_cap)
        phi, _ = fit_order(pp)
        return enforce_stationary_ar(phi, 0.99)
    nb = r_bic_n(mat, n, vbar=pooled.num[0])
    best_bic = nb * np.log(max(gamma0[0], 1e-12)) + np.log(nb)
    best = np.array([])
    p_sel = min(p_cap, n // 5)
    for pp in range(1, p_sel + 1):
        phi, s2 = fit_order(pp)
        bic = nb * np.log(s2) + (pp + 1) * np.log(nb)
        if not np.isfinite(bic) or bic >= best_bic:
            continue
        ph = enforce_stationary_ar(phi, 0.99)
        if ph.size != pp:
            continue
        best_bic, best = bic, ph
    return best


# --- R .solve_correction (fit_and_apply.R, 5f5fc32 + bd4e03f) ---------------
def r_solve_correction(A, g, sv_tol=0.1, tail_frac=0.25):
    L = A.shape[1]
    if L < 8:
        return np.linalg.solve(A, g)
    U, d, Vt = np.linalg.svd(A)
    keep = d > sv_tol * np.median(d)
    if np.all(keep):
        return np.linalg.solve(A, g)
    V = Vt.T
    x0 = V[:, keep] @ ((U[:, keep].T @ g) / d[keep])
    N = V[:, ~keep]
    tl = np.arange(L - int(np.ceil(tail_frac * L)), L)
    cc = np.linalg.lstsq(N[tl], -x0[tl], rcond=None)[0]
    return x0 + N @ cc


def r_apply_acvf_correction(gamma, A):
    """R .apply_acvf_correction_result with .solve_correction in place."""
    gamma = np.asarray(gamma, float).ravel()
    A = np.asarray(A, float)
    L = min(gamma.size, A.shape[0])
    if L < 1:
        return gamma, False
    Asub = A[:L, :L]
    rc = 1.0 / np.linalg.cond(Asub, 1)
    if not np.isfinite(rc) or rc < 1e-6:
        return gamma, False
    try:
        out = r_solve_correction(Asub, gamma[:L])
    except np.linalg.LinAlgError:
        return gamma, False
    if not np.all(np.isfinite(out)) or out[0] <= 0:
        return gamma, False
    if gamma.size > L:
        out = np.concatenate([out, gamma[L:]])
    return out, True


# --- exact GLS oracle (test-audit-regressions.R gls_oracle) ------------------
def arma_acvf_unit(phi, theta, max_lag, nmA=5000):
    """Unit-innovation ARMA autocovariance via the MA(inf) expansion."""
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    imp = np.zeros(nmA)
    imp[0] = 1.0
    psi = lfilter(np.concatenate([[1.0], theta]), np.concatenate([[1.0], -phi]), imp)
    return np.array([np.dot(psi[: nmA - k], psi[k:]) for k in range(max_lag + 1)])


def gls_oracle(y, phi, theta):
    n = len(y)
    g = arma_acvf_unit(phi, theta, n - 1)
    S = g[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    Lc = np.linalg.cholesky(S)
    return np.linalg.solve(Lc, y)
