"""Run- and censor-aware noise autocovariance.

Ports fmriAR 0.3.3 ``noise_acvf.R``, ``.pooled_acvf_segments``,
``.acvf_from_pooled``, and residual-bias correction (``acvf_bias.R``).

Lag products never cross a run boundary or a censoring gap. The mean is
removed per run, not per fragment, so two-frame scrubbing fragments do
not force lag-1 correlation to -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
import warnings

import numpy as np
from numpy.linalg import eigvalsh
from numpy.typing import NDArray

from .numhelpers import enforce_stationary_ar, levinson_durbin

ACVF_RCOND_MIN = 1e-6


@dataclass
class PooledAcvf:
    """Per-lag product sums (column-averaged) and pair counts."""

    num: NDArray[np.float64]
    pairs: NDArray[np.float64]


@dataclass
class ValidSegments:
    """Non-censored timepoints and the contiguous segments they form."""

    idx: NDArray[np.intp]
    starts0: NDArray[np.intp]
    run_id: NDArray[np.intp]


@dataclass
class NoiseAcvf:
    """Exported autocovariance estimate (``fmriAR_acvf``)."""

    acvf: dict[str, NDArray[np.float64]]
    pairs: dict[str, NDArray[np.float64]]
    n_segments: dict[str, int]
    segment_lengths: dict[str, NDArray[np.intp]]
    max_lag: int
    pooling: str
    corrected: bool = False


def run_codes(
    runs: Optional[NDArray[Any]], n: int, arg: str = "runs"
) -> NDArray[np.intp]:
    """Integer run codes, one contiguous block per label."""
    n = int(n)
    if runs is None:
        return np.ones(n, dtype=np.intp)
    runs_arr = np.asarray(runs)
    if runs_arr.shape[0] != n:
        raise ValueError(
            f"'{arg}' has length {runs_arr.shape[0]} but must have one "
            f"entry per timepoint ({n})"
        )
    if np.any(runs_arr != runs_arr):  # NA for numeric; object NA handled below
        raise ValueError(f"'{arg}' contains NA")
    if runs_arr.dtype == object or np.issubdtype(runs_arr.dtype, np.str_):
        if np.any(runs_arr == None):  # noqa: E711
            raise ValueError(f"'{arg}' contains NA")
    uniq: list[Any] = []
    codes = np.empty(n, dtype=np.intp)
    seen: dict[Any, int] = {}
    for i, lab in enumerate(runs_arr.tolist()):
        if lab not in seen:
            seen[lab] = len(uniq) + 1
            uniq.append(lab)
        codes[i] = seen[lab]
    # Contiguous-block check: rle of codes must have unique values.
    changes = np.concatenate([[True], np.diff(codes) != 0])
    blocks = codes[changes]
    if len(blocks) != len(np.unique(blocks)):
        raise ValueError(f"each '{arg}' label must occupy one contiguous block")
    return codes


def run_sets(
    runs: Optional[NDArray[Any]], n: int, arg: str = "runs"
) -> list[tuple[str, NDArray[np.intp]]]:
    """Named (label, 0-based indices) run split in time order."""
    codes = run_codes(runs, n, arg=arg)
    if runs is None:
        return [("1", np.arange(n, dtype=np.intp))]
    labels = np.asarray(runs)
    out: list[tuple[str, NDArray[np.intp]]] = []
    for code in np.unique(codes):
        idx = np.where(codes == code)[0]
        lab = labels[idx[0]]
        out.append((str(lab), idx.astype(np.intp, copy=False)))
    return out


def parcel_codes(x: NDArray[Any], arg: str = "parcels") -> NDArray[np.intp]:
    """Validate parcel labels; refuse NA, fractional, and non-integerable."""
    arr = np.asarray(x)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
        sample = ", ".join(repr(v) for v in list(np.unique(arr))[:3])
        raise ValueError(
            f"'{arg}' must be integer, numeric, or factor labels; "
            f"character labels cannot be coerced ({sample}). Convert explicitly "
            f"with integer codes (e.g. as.integer(factor({arg}))) and reuse "
            "the same coding when applying the plan."
        )
    if np.issubdtype(arr.dtype, np.floating):
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"'{arg}' contains NA")
        if np.any(arr != np.trunc(arr)):
            raise ValueError(f"'{arg}' must contain finite whole-number parcel labels")
    try:
        codes = np.asarray(arr, dtype=np.intp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{arg}' must be integer, numeric, or factor labels") from exc
    return codes


def valid_segments(
    n: int,
    runs: Optional[NDArray[Any]] = None,
    censor: Optional[NDArray[Any]] = None,
) -> ValidSegments:
    """Non-censored indices; segments break at run boundaries and gaps."""
    n = int(n)
    r = run_codes(runs, n)
    valid = np.ones(n, dtype=bool)
    if censor is not None and len(np.asarray(censor)):
        c = np.asarray(censor, dtype=np.intp)
        c = c[(c >= 0) & (c < n)]
        valid[c] = False
    idx = np.where(valid)[0].astype(np.intp, copy=False)
    if idx.size == 0:
        return ValidSegments(
            idx=idx,
            starts0=np.array([], dtype=np.intp),
            run_id=np.array([], dtype=np.intp),
        )
    brk = np.ones(idx.size, dtype=bool)
    if idx.size > 1:
        brk[1:] = (np.diff(idx) != 1) | (r[idx[1:]] != r[idx[:-1]])
    starts0 = np.where(brk)[0].astype(np.intp, copy=False)
    return ValidSegments(idx=idx, starts0=starts0, run_id=r[idx].astype(np.intp))


def pooled_acvf_segments(
    mat: NDArray[np.float64],
    seg_id: NDArray[Any],
    max_lag: int,
    center_id: Optional[NDArray[Any]] = None,
) -> PooledAcvf:
    """Pool lag products within segments; mean comes from ``center_id``."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    max_lag = max(0, int(max_lag))
    nv, nc = mat.shape
    num = np.zeros(max_lag + 1, dtype=np.float64)
    pairs = np.zeros(max_lag + 1, dtype=np.float64)
    if nv == 0 or nc == 0:
        return PooledAcvf(num=num, pairs=pairs)

    if center_id is None:
        mat = mat - mat.mean(axis=0, keepdims=True)
    else:
        cid = np.asarray(center_id)
        uniq, inv = np.unique(cid, return_inverse=True)
        mu = np.zeros((len(uniq), nc), dtype=np.float64)
        counts = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(mu, inv, mat)
        np.add.at(counts, inv, 1.0)
        mu /= np.maximum(counts[:, np.newaxis], 1.0)
        mat = mat - mu[inv]

    seg = np.asarray(seg_id)
    num[0] = float(np.sum(mat * mat) / nc)
    pairs[0] = float(nv)
    for lg in range(1, max_lag + 1):
        if nv <= lg:
            break
        hi = np.arange(lg, nv)
        lo = np.arange(0, nv - lg)
        ok = seg[hi] == seg[lo]
        if not np.any(ok):
            continue
        num[lg] = float(np.sum(mat[hi[ok]] * mat[lo[ok]]) / nc)
        pairs[lg] = float(np.sum(ok))
    return PooledAcvf(num=num, pairs=pairs)


def acvf_max_lag(pooled: PooledAcvf) -> int:
    usable = np.where(pooled.pairs > 0)[0]
    if usable.size == 0:
        return -1
    return int(usable.max())


def acvf_is_psd(gamma: NDArray[np.float64], tol: float = 1e-6) -> bool:
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    if gamma.size == 0 or not np.all(np.isfinite(gamma)) or gamma[0] <= 0:
        return False
    if gamma.size < 2:
        return True
    toeplitz = np.empty((gamma.size, gamma.size), dtype=np.float64)
    for i in range(gamma.size):
        for j in range(gamma.size):
            toeplitz[i, j] = gamma[abs(i - j)]
    try:
        ev = eigvalsh(toeplitz)
    except np.linalg.LinAlgError:
        return False
    return bool(np.min(ev) >= tol * gamma[0])


def shrink_to_pd(gamma: NDArray[np.float64], tol: float = 1e-6) -> NDArray[np.float64]:
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    if gamma.size == 0:
        return gamma
    if acvf_is_psd(gamma, tol):
        return gamma
    lo, hi = 0.0, 1.0
    g0 = gamma[0]
    tail = gamma[1:]
    for _ in range(50):
        mid = (lo + hi) / 2.0
        trial = np.concatenate([[g0], tail * mid])
        if acvf_is_psd(trial, tol):
            lo = mid
        else:
            hi = mid
    return np.concatenate([[g0], tail * lo])


def apply_acvf_correction(
    gamma: NDArray[np.float64], A: NDArray[np.float64]
) -> tuple[NDArray[np.float64], bool]:
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    L = min(gamma.size, A.shape[0])
    if L < 1:
        return gamma, False
    Asub = A[:L, :L]
    try:
        rc = float(np.linalg.cond(Asub))
        rcond = (1.0 / rc) if np.isfinite(rc) and rc > 0 else 0.0
    except np.linalg.LinAlgError:
        return gamma, False
    if not np.isfinite(rcond) or rcond < ACVF_RCOND_MIN:
        return gamma, False
    try:
        out = np.linalg.solve(Asub, gamma[:L])
    except np.linalg.LinAlgError:
        return gamma, False
    if not np.all(np.isfinite(out)) or out[0] <= 0:
        return gamma, False
    if gamma.size > L:
        out = np.concatenate([out, gamma[L:]])
    return out.astype(np.float64, copy=False), True


def acvf_from_pooled(
    pooled: PooledAcvf,
    order: Optional[int] = None,
    tol: float = 1e-6,
    correction: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], bool]:
    """Unbiased ACVF with optional bias correction, then PSD shrink."""
    num = pooled.num
    pairs = pooled.pairs
    usable = np.where(pairs > 0)[0]
    if usable.size == 0:
        return np.array([], dtype=np.float64), False
    avail = int(usable.max()) + 1
    if pairs[0] <= 0:
        return np.array([], dtype=np.float64), False
    keep_n = avail if order is None else max(1, min(int(order) + 1, avail))
    g_unb = np.zeros(avail, dtype=np.float64)
    pos = pairs[:avail] > 0
    g_unb[pos] = num[:avail][pos] / pairs[:avail][pos]
    if not np.isfinite(g_unb[0]) or g_unb[0] <= 0:
        return np.array([], dtype=np.float64), False

    applied = False
    if correction is not None:
        g_unb, applied = apply_acvf_correction(g_unb, correction)
        if g_unb.size == 0 or not np.isfinite(g_unb[0]) or g_unb[0] <= 0:
            return np.array([], dtype=np.float64), False

    g_unb = g_unb[:keep_n]
    if g_unb.size == 0:
        return np.array([], dtype=np.float64), False
    return shrink_to_pd(g_unb, tol), applied


def _apply_lag_operator(
    B: NDArray[np.float64], k: int, run_vec: NDArray[np.intp]
) -> NDArray[np.float64]:
    if k == 0:
        return B
    n = B.shape[1]
    out = np.zeros_like(B)
    if k >= n:
        return out
    hi = np.arange(k, n)
    lo = np.arange(0, n - k)
    same = run_vec[hi] == run_vec[lo]
    if not np.any(same):
        return out
    hi_s = hi[same]
    lo_s = lo[same]
    out[:, hi_s] += B[:, lo_s]
    out[:, lo_s] += B[:, hi_s]
    return out


def acvf_bias_core(
    Q: NDArray[np.float64],
    run_vec: NDArray[np.intp],
    idx: NDArray[np.intp],
    seg_id: NDArray[Any],
    max_lag: int,
) -> NDArray[np.float64]:
    nv = int(idx.size)
    max_lag = max(0, int(max_lag))
    if nv < 2:
        return np.eye(max_lag + 1)
    R = -Q[idx] @ Q.T
    R[np.arange(nv), idx] += 1.0
    R = R - R.mean(axis=0, keepdims=True)

    a_list: list[NDArray[np.intp]] = []
    b_list: list[NDArray[np.intp]] = []
    npair = np.zeros(max_lag + 1, dtype=np.float64)
    a_list.append(np.arange(nv, dtype=np.intp))
    b_list.append(np.arange(nv, dtype=np.intp))
    npair[0] = float(nv)
    seg = np.asarray(seg_id)
    for lg in range(1, max_lag + 1):
        if nv <= lg:
            a_list.append(np.array([], dtype=np.intp))
            b_list.append(np.array([], dtype=np.intp))
            break
        hi = np.arange(lg, nv)
        lo = np.arange(0, nv - lg)
        ok = seg[hi] == seg[lo]
        a_list.append(hi[ok].astype(np.intp, copy=False))
        b_list.append(lo[ok].astype(np.intp, copy=False))
        npair[lg] = float(np.sum(ok))

    A = np.eye(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        SkR = _apply_lag_operator(R, k, run_vec)
        for h in range(max_lag + 1):
            if npair[h] <= 0 or a_list[h].size == 0:
                continue
            A[h, k] = float(np.sum(R[a_list[h]] * SkR[b_list[h]]) / npair[h])
    return A


def acvf_bias_matrix(
    design: NDArray[np.float64],
    runs: Optional[NDArray[Any]] = None,
    censor: Optional[NDArray[Any]] = None,
    max_lag: int = 25,
) -> list[NDArray[np.float64]]:
    """Bias matrices ``E[gamma_raw] = A gamma_true``, one per run."""
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, np.newaxis]
    n = design.shape[0]
    if np.any(~np.isfinite(design)):
        raise ValueError("'design' contains NA")
    run_vec = run_codes(runs, n)
    q, r = np.linalg.qr(design, mode="reduced")
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-12 * np.max(np.abs(np.diag(r)))))
    Q = q[:, : max(rank, 1)]
    df = n - Q.shape[1]
    max_lag = int(max_lag)
    if df < max_lag + 1:
        capped = max(1, df - 1)
        warnings.warn(
            f"residual-bias correction: the design leaves {df} residual "
            f"degrees of freedom, fewer than the {max_lag + 1} autocovariance "
            f"lags requested. Reducing the correction lag budget to {capped}.",
            UserWarning,
            stacklevel=2,
        )
        max_lag = capped

    sets = run_sets(runs, n)
    cens = (
        np.array([], dtype=np.intp)
        if censor is None
        else np.asarray(censor, dtype=np.intp)
    )
    out: list[NDArray[np.float64]] = []
    for _, ridx in sets:
        keep = np.setdiff1d(ridx, cens, assume_unique=False)
        if keep.size < 2:
            out.append(np.eye(max_lag + 1))
            continue
        seg_id = np.cumsum(np.concatenate([[1], (np.diff(keep) > 1).astype(np.intp)]))
        out.append(acvf_bias_core(Q, run_vec, keep, seg_id, max_lag))
    return out


def drop_unusable_corrections(
    corr_by_run: Sequence[Optional[NDArray[np.float64]]],
) -> list[Optional[NDArray[np.float64]]]:
    out: list[Optional[NDArray[np.float64]]] = []
    for i, A in enumerate(corr_by_run):
        if A is None:
            out.append(None)
            continue
        A = np.asarray(A, dtype=np.float64)
        try:
            rc = float(np.linalg.cond(A))
            rcond = (1.0 / rc) if np.isfinite(rc) and rc > 0 else 0.0
        except np.linalg.LinAlgError:
            rcond = 0.0
        if not np.isfinite(rcond) or rcond < ACVF_RCOND_MIN:
            warnings.warn(
                f"residual-bias correction skipped for run {i + 1}: "
                f"the correction matrix is ill-conditioned (rcond = {rcond:.3g}) "
                f"at a lag budget of {A.shape[0] - 1}. Lower "
                "'correction_max_lag'. Estimates for this run are uncorrected.",
                UserWarning,
                stacklevel=2,
            )
            out.append(None)
        else:
            out.append(A)
    return out


def yw_from_acvf(
    gamma: NDArray[np.float64], p: int
) -> tuple[NDArray[np.float64], float]:
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    p = int(p)
    if p < 1 or gamma.size < p + 1:
        return np.array([], dtype=np.float64), float("nan")
    return levinson_durbin(gamma[: p + 1], p)


def estimate_ar_series(
    y: NDArray[np.float64],
    p_max: int,
    p: object = "auto",
    starts0: Optional[NDArray[Any]] = None,
    center_id: Optional[NDArray[Any]] = None,
) -> dict[str, Any]:
    """Order selection / fixed-p fit for one series with segment structure."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if starts0 is None or len(np.asarray(starts0)) == 0:
        starts0_arr = np.array([0], dtype=np.intp)
    else:
        starts0_arr = np.asarray(starts0, dtype=np.intp)
    empty: dict[str, Any] = {
        "phi": np.array([], dtype=np.float64),
        "order": (0, 0),
        "gamma": np.array([], dtype=np.float64),
        "sigma2": float("nan"),
    }
    ends = np.concatenate([starts0_arr[1:], [n]])
    seg_len = ends - starts0_arr
    p_cap = min(int(p_max), int(np.max(seg_len) - 1) if seg_len.size else -1, n - 1)
    if p_cap < 1:
        return empty
    is_start = np.zeros(n, dtype=bool)
    is_start[starts0_arr[starts0_arr < n]] = True
    is_start[0] = True
    seg_id = np.cumsum(is_start.astype(np.intp))
    pooled = pooled_acvf_segments(y[:, np.newaxis], seg_id, p_cap, center_id=center_id)
    gamma0, _ = acvf_from_pooled(pooled, order=0)
    if gamma0.size == 0 or not np.isfinite(gamma0[0]) or gamma0[0] <= 0:
        return empty
    p_cap = min(p_cap, acvf_max_lag(pooled))
    if p_cap < 1:
        empty["gamma"] = gamma0
        empty["sigma2"] = float(gamma0[0])
        return empty
    gamma_full, _ = acvf_from_pooled(pooled, order=p_cap)

    def fit_order(pp: int) -> tuple[NDArray[np.float64], float]:
        g, _ = acvf_from_pooled(pooled, order=pp)
        phi, sigma2 = yw_from_acvf(g[: pp + 1], pp)
        return enforce_stationary_ar(phi, 0.99), float(max(sigma2, 1e-12))

    if p != "auto":
        pp = min(int(p), p_cap)
        if pp <= 0:
            empty["gamma"] = gamma_full
            empty["sigma2"] = float(gamma_full[0]) if gamma_full.size else float("nan")
            return empty
        phi, sigma2 = fit_order(pp)
        return {
            "phi": phi,
            "order": (pp, 0),
            "gamma": gamma_full,
            "sigma2": sigma2,
        }

    n_eff = float(n)
    n_eff_log = np.log(n_eff)
    sigma0 = max(float(gamma0[0]), 1e-12)
    best_bic = 2.0 * n_eff * np.log(sigma0) + n_eff_log
    best_phi = np.array([], dtype=np.float64)
    best_p = 0
    best_sigma2 = sigma0
    p_sel = min(p_cap, int(np.floor(n_eff / 5.0)))
    for pp in range(1, max(p_sel, 0) + 1):
        phi_pp, sigma2 = fit_order(pp)
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
        "order": (best_p, 0),
        "gamma": gamma_full,
        "sigma2": best_sigma2,
    }


def sigma2_from_gamma_phi(
    gamma: NDArray[np.float64], phi: NDArray[np.float64]
) -> float:
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if gamma.size == 0 or not np.isfinite(gamma[0]) or gamma[0] <= 0:
        return float("nan")
    if phi.size == 0:
        return float(gamma[0])
    if not np.all(np.isfinite(phi)):
        return float("nan")
    if gamma.size < phi.size + 1:
        return float("nan")
    s2 = float(gamma[0] - np.dot(phi, gamma[1 : phi.size + 1]))
    return float(min(max(s2, 1e-12), gamma[0]))


def noise_acvf(
    resid: NDArray[np.float64],
    runs: Optional[NDArray[Any]] = None,
    censor: Optional[NDArray[Any]] = None,
    max_lag: int = 20,
    pooling: str = "global",
    parcels: Optional[NDArray[Any]] = None,
    design: Optional[NDArray[np.float64]] = None,
    correction_max_lag: int = 25,
) -> NoiseAcvf:
    """Run- and censor-aware noise autocovariance."""
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    n = resid.shape[0]
    if n < 2:
        raise ValueError("'resid' needs at least two timepoints")
    if pooling not in ("global", "run", "parcel"):
        raise ValueError("pooling must be 'global', 'run', or 'parcel'")
    if np.any(~np.isfinite(resid)):
        raise ValueError("'resid' contains NA, NaN, or Inf")
    max_lag = int(min(max(max_lag, 0), n))

    if censor is not None:
        c = np.asarray(censor)
        if c.dtype == bool:
            c = np.where(c)[0]
        censor = np.unique(np.asarray(c, dtype=np.intp))
        censor = censor[(censor >= 0) & (censor < n)]
        if censor.size == 0:
            censor = None

    corr_by_run: Optional[list[Optional[NDArray[np.float64]]]] = None
    if design is not None:
        mats = acvf_bias_matrix(
            design, runs=runs, censor=censor, max_lag=int(min(correction_max_lag, n))
        )
        corr_by_run = drop_unusable_corrections(mats)

    sets = run_sets(runs, n)
    if pooling == "parcel":
        if parcels is None:
            raise ValueError("'parcels' is required when pooling = 'parcel'")
        parcels_c = parcel_codes(parcels)
        if parcels_c.size != resid.shape[1]:
            raise ValueError("parcels must have one entry per voxel")
        if design is not None:
            raise ValueError(
                "residual-bias correction is not yet supported for pooling='parcel'"
            )
        seg = valid_segments(n, runs=runs, censor=censor)
        if seg.idx.size < 2:
            raise ValueError("no valid segments remain after censoring")
        is_start = np.zeros(seg.idx.size, dtype=bool)
        if seg.idx.size:
            is_start[seg.starts0[seg.starts0 < seg.idx.size]] = True
            is_start[0] = True
        seg_id = np.cumsum(is_start.astype(np.intp))
        units: dict[str, dict[str, Any]] = {}
        for pid in np.unique(parcels_c):
            cols = np.where(parcels_c == pid)[0]
            if cols.size == 0:
                continue
            pooled = pooled_acvf_segments(
                resid[np.ix_(seg.idx, cols)],
                seg_id,
                max_lag,
                center_id=seg.run_id,
            )
            g, _ = acvf_from_pooled(pooled, order=max_lag)
            if g.size == 0:
                continue
            key = str(int(pid))
            units[key] = {
                "acvf": g,
                "pairs": pooled.pairs[: g.size],
                "n_seg": int(seg_id.max()) if seg_id.size else 0,
                "seg_len": np.array(
                    [int(np.sum(seg_id == s)) for s in np.unique(seg_id)],
                    dtype=np.intp,
                ),
                "corrected": False,
            }
    else:
        units = {}
        all_cols = np.arange(resid.shape[1])
        for i, (lab, idx) in enumerate(sets):
            keep = idx if censor is None else np.setdiff1d(idx, censor)
            if keep.size < 2:
                continue
            seg_id = np.cumsum(
                np.concatenate([[1], (np.diff(keep) > 1).astype(np.intp)])
            )
            corr = None
            if corr_by_run is not None:
                corr = corr_by_run[i]
            lag_budget = max_lag if corr is None else max(max_lag, corr.shape[0] - 1)
            pooled = pooled_acvf_segments(
                resid[np.ix_(keep, all_cols)], seg_id, lag_budget
            )
            g, applied = acvf_from_pooled(pooled, order=max_lag, correction=corr)
            if g.size == 0:
                continue
            units[lab] = {
                "acvf": g,
                "pairs": pooled.pairs[: g.size],
                "n_seg": int(seg_id.max()) if seg_id.size else 0,
                "seg_len": np.array(
                    [int(np.sum(seg_id == s)) for s in np.unique(seg_id)],
                    dtype=np.intp,
                ),
                "corrected": applied,
            }

    if not units:
        raise ValueError("no valid segments remain after censoring")

    if pooling == "global" and len(units) > 1:
        L = min(u["acvf"].size for u in units.values())
        w = np.array([float(np.sum(u["seg_len"])) for u in units.values()])
        w = w / w.sum()
        G = np.vstack([u["acvf"][:L] for u in units.values()])
        P = np.vstack([u["pairs"][:L] for u in units.values()])
        all_corr = all(bool(u["corrected"]) for u in units.values())
        units = {
            "1": {
                "acvf": w @ G,
                "pairs": P.sum(axis=0),
                "n_seg": sum(int(u["n_seg"]) for u in units.values()),
                "seg_len": np.concatenate([u["seg_len"] for u in units.values()]),
                "corrected": all_corr,
            }
        }
    if pooling == "global":
        only = next(iter(units.values()))
        units = {"1": only}

    return NoiseAcvf(
        acvf={k: np.asarray(v["acvf"], dtype=np.float64) for k, v in units.items()},
        pairs={k: np.asarray(v["pairs"], dtype=np.float64) for k, v in units.items()},
        n_segments={k: int(v["n_seg"]) for k, v in units.items()},
        segment_lengths={
            k: np.asarray(v["seg_len"], dtype=np.intp) for k, v in units.items()
        },
        max_lag=max_lag,
        pooling=pooling,
        corrected=all(bool(v["corrected"]) for v in units.values()),
    )
