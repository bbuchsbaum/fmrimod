"""Run- and censor-aware noise autocovariance.

Ports fmriAR 0.3.3 ``noise_acvf.R``, ``.pooled_acvf_segments``,
``.acvf_from_pooled``, and residual-bias correction (``acvf_bias.R``).

Lag products never cross a run boundary or a censoring gap. The mean is
removed per run, not per fragment, so two-frame scrubbing fragments do
not force lag-1 correlation to -1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import warnings

import numpy as np
from numpy.linalg import eigvalsh
from numpy.typing import NDArray
from scipy import linalg as scipy_linalg

from .numhelpers import enforce_stationary_ar, levinson_durbin

ACVF_RCOND_MIN = 1e-6
# R qr()'s default rank tolerance.
_QR_RANK_TOL = 1e-7
# fmriAR .solve_correction heuristics, ported verbatim (5f5fc32, bd4e03f):
# directions with a singular value below this fraction of the MEDIAN are
# pinned (the median, because a design leaving few residual df shrinks the
# whole spectrum relative to the largest value), and the pin makes this
# trailing fraction of the corrected lag budget as close to zero as possible.
_CORRECTION_SV_TOL = 0.1
_CORRECTION_TAIL_FRAC = 0.25


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


def _normalize_censor(
    censor: NDArray[Any] | None, n: int, arg: str = "censor"
) -> NDArray[np.intp] | None:
    """Sorted unique 0-based censored frames inside ``[0, n)``, or ``None``.

    A boolean mask is converted with ``flatnonzero``: casting it to integers
    (fmriAR 617bd87) censors frames 0 and 1 instead of the flagged ones. A
    mask must have one entry per timepoint; indices must be finite whole
    numbers (``2.5`` is refused rather than truncated). Missing values
    (``None``/NaN) are refused in either form; out-of-range indices are
    dropped.
    """
    if censor is None:
        return None
    c = np.asarray(censor)
    if c.size == 0:
        return None
    if c.dtype == object:
        items = c.ravel().tolist()
        if any(v is None for v in items):
            raise ValueError(f"'{arg}' contains NA")
        if all(isinstance(v, (bool, np.bool_)) for v in items):
            c = c.astype(bool)
    if c.dtype == bool:
        if c.size != n:
            raise ValueError(
                f"logical '{arg}' must have one entry per timepoint ({n}), "
                f"not {c.size}"
            )
        idx = np.flatnonzero(c)
    else:
        try:
            cf = np.asarray(c, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            cf = np.array([np.nan])
        if np.any(~np.isfinite(cf)) or np.any(cf != np.trunc(cf)):
            raise ValueError(
                f"'{arg}' must be whole-number timepoint indices or a "
                "boolean mask"
            )
        idx = np.unique(cf.astype(np.intp))
    idx = idx[(idx >= 0) & (idx < n)].astype(np.intp, copy=False)
    return idx if idx.size else None


def valid_segments(
    n: int,
    runs: Optional[NDArray[Any]] = None,
    censor: Optional[NDArray[Any]] = None,
) -> ValidSegments:
    """Non-censored indices; segments break at run boundaries and gaps."""
    n = int(n)
    r = run_codes(runs, n)
    valid = np.ones(n, dtype=bool)
    c = _normalize_censor(censor, n)
    if c is not None:
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
    mat = np.ascontiguousarray(mat)
    num[0] = float(np.vdot(mat, mat) / nc)
    pairs[0] = float(nv)
    for lg in range(1, max_lag + 1):
        if nv <= lg:
            break
        ok = seg[lg:] == seg[:-lg]
        n_ok = int(np.sum(ok))
        if n_ok == 0:
            continue
        # One dot product per run of consecutive usable pairs, on contiguous
        # row slices, so no full-size copy is made per lag.
        edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
        total = 0.0
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1),
                        strict=True):
            total += float(np.vdot(mat[a + lg : b + lg], mat[a:b]))
        num[lg] = total / nc
        pairs[lg] = float(n_ok)
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


def _qr_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``qr.solve``: least squares that refuses a rank-deficient system.

    An underdetermined full-row-rank system gets R's basic solution, with
    the trailing coefficients set to zero.
    """
    nr, nc = a.shape
    d = np.abs(np.diag(scipy_linalg.qr(a, mode="r", pivoting=True)[0]))
    rank = int(np.sum(d > _QR_RANK_TOL * d[0])) if d.size and d[0] > 0 else 0
    if rank != min(nr, nc):
        raise np.linalg.LinAlgError("singular matrix 'a' in solve")
    if nr >= nc:
        return np.asarray(np.linalg.lstsq(a, b, rcond=None)[0], dtype=np.float64)
    x = np.zeros(nc, dtype=np.float64)
    x[:nr] = np.linalg.solve(a[:, :nr], b)
    return x


def _solve_correction(
    A: NDArray[np.float64],
    g: NDArray[np.float64],
    sv_tol: float = _CORRECTION_SV_TOL,
    tail_frac: float = _CORRECTION_TAIL_FRAC,
) -> NDArray[np.float64]:
    """Solve ``A gamma = g``, pinning the directions the residuals cannot see.

    Projecting out run means and slow drift leaves ``A`` with near-null
    directions (about a constant offset across lags). Without censoring
    the raw estimate carries almost no noise along them, but censoring
    breaks that, and the exact solve returned phi with sd ~0.3 for a truth
    of 0.4. Directions with a singular value below ``sv_tol`` times the
    median are not taken from the data: their coefficients make the last
    ``tail_frac`` of the corrected autocovariance as close to zero as
    possible, the short-memory assumption the truncated correction already
    makes. Well-determined directions are solved exactly, so a
    well-conditioned map gets ``solve(A, g)`` (fmriAR 5f5fc32, bd4e03f
    ``.solve_correction``).
    """
    L = A.shape[1]
    if L < 8:
        return np.asarray(np.linalg.solve(A, g), dtype=np.float64)
    U, d, Vt = np.linalg.svd(A)
    keep = d > sv_tol * np.median(d)
    if np.all(keep):
        return np.asarray(np.linalg.solve(A, g), dtype=np.float64)
    V = Vt.T
    x0 = V[:, keep] @ ((U[:, keep].T @ g) / d[keep])
    N = V[:, ~keep]
    tail = slice(L - int(np.ceil(tail_frac * L)), L)
    return np.asarray(x0 + N @ _qr_solve(N[tail], -x0[tail]), dtype=np.float64)


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
        out = _solve_correction(Asub, gamma[:L])
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


class DesignResidualMismatchError(ValueError):
    """``resid`` is not OLS-residualised by the ``design`` given for correction.

    fmriAR's condition class ``fmriAR_design_residual_mismatch`` (ca27f27).
    """


def _validate_design_residuals(
    resid: NDArray[np.float64], design: NDArray[np.float64], tol: float = 1e-6
) -> float:
    """Refuse residuals that are not numerically orthogonal to ``design``.

    The design correction is derived for OLS residuals ``(I - QQ') y``; any
    other residual operator (a raw series, WLS) asks it to undo a projection
    that did not occur. Orthogonality is necessary, not sufficient: residuals
    of a nested design also pass, so provenance stays the caller's job
    (fmriAR ``.validate_design_residuals``). The test is relative to the
    residual norm and uses the rank-revealing column space, so rescaling,
    reordering or aliasing design columns cannot change the verdict.
    """
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, np.newaxis]
    if design.shape[0] != resid.shape[0]:
        raise ValueError(
            f"'design' must have one row per timepoint in 'resid' "
            f"({resid.shape[0]}), not {design.shape[0]}"
        )
    if np.any(~np.isfinite(design)):
        raise ValueError("'design' contains NA, NaN, or Inf")
    if design.size == 0 or resid.size == 0:
        return 0.0
    q, r, _ = scipy_linalg.qr(design, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int(np.sum(d > _QR_RANK_TOL * d[0])) if d.size and d[0] > 0 else 0
    resid_norm = float(np.linalg.norm(resid))
    if rank == 0 or not np.isfinite(resid_norm) or resid_norm == 0:
        return 0.0
    rel = float(np.linalg.norm(q[:, :rank].T @ resid)) / resid_norm
    if not np.isfinite(rel) or rel > tol:
        raise DesignResidualMismatchError(
            f"'resid' is not orthogonal to 'design' (relative projection "
            f"{rel:.3g}). Design-based ACVF correction requires residuals "
            "numerically compatible with OLS residualization by this design; "
            "omit 'design' or supply the matching OLS residuals."
        )
    return rel


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
    # The map depends only on the column space, so take Q[:, :rank] from a
    # pivoted QR (R qr(), dqrdc2). Without pivoting a dependent column that is
    # not last puts an arbitrary direction into the leading rank columns.
    q, r, _ = scipy_linalg.qr(design, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int(np.sum(d > _QR_RANK_TOL * d[0])) if d.size and d[0] > 0 else 0
    Q = q[:, :rank]
    df = n - rank
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
    cens = _normalize_censor(censor, n)
    if cens is None:
        cens = np.array([], dtype=np.intp)
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


def _bic_n(
    mat: NDArray[np.float64],
    n_frames: float,
    groups: NDArray[Any] | None = None,
    vbar: float | None = None,
) -> float:
    """BIC sample size of a pooled fit: frames times effective voxels.

    The effective count is the Kish design effect ``V / (1 + (V - 1) rbar)``,
    with the mean inter-voxel correlation ``rbar`` read off the variance of
    the voxel mean (fmriAR 61b89ff ``.bic_n``). Frames alone treat pooled
    voxels as one series and under-select; every voxel over-selects when
    they share fluctuations. ``groups`` labels rows centred separately
    (runs); ``vbar`` is the pooled lag-0 sum the caller already holds.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    n_rows, V = mat.shape
    if V <= 1 or n_rows < 2:
        return float(n_frames)
    if groups is None:
        inv = np.zeros(n_rows, dtype=np.intp)
    else:
        inv = np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    counts = np.bincount(inv).astype(np.float64)
    rm = mat.mean(axis=1)
    rm_c = rm - (np.bincount(inv, weights=rm) / counts)[inv]
    vmean = float(np.dot(rm_c, rm_c))
    if vbar is None:
        mu = np.zeros((counts.size, V), dtype=np.float64)
        np.add.at(mu, inv, mat)
        centred = mat - (mu / counts[:, np.newaxis])[inv]
        vbar = float(np.sum(centred * centred)) / V
    if not vbar > 0:
        return float(n_frames)
    rbar = min(max((V * vmean / vbar - 1.0) / (V - 1.0), 0.0), 1.0)
    return float(n_frames * max(1.0, min(V, V / (1.0 + (V - 1.0) * rbar))))


def _select_ar_order(
    fit_order: Callable[[int], tuple[NDArray[np.float64], float]],
    gamma0: float,
    p_sel: int,
    n_bic: float,
) -> tuple[NDArray[np.float64], int, float]:
    """BIC order search shared by the run, global and parcel estimators.

    ``BIC = N log(sigma2) + k log(N)`` with ``N = n_bic``: -2 log L of a
    Gaussian AR is ``N log(sigma2)`` up to a constant, and the doubled fit
    term ported from fmriAR 0.3.3 halved the penalty (fmriAR 617bd87).
    ``fit_order(p)`` returns the stationarity-enforced coefficients and the
    floored innovation variance. Returns ``(phi, p, sigma2)``.
    """
    log_n = float(np.log(n_bic))
    sigma0 = max(float(gamma0), 1e-12)
    best_bic = n_bic * np.log(sigma0) + log_n
    best: tuple[NDArray[np.float64], int, float] = (
        np.array([], dtype=np.float64),
        0,
        sigma0,
    )
    for pp in range(1, max(int(p_sel), 0) + 1):
        phi, sigma2 = fit_order(pp)
        if not np.isfinite(sigma2) or phi.size != pp or not np.all(np.isfinite(phi)):
            continue
        bic = n_bic * np.log(sigma2) + (pp + 1) * log_n
        if np.isfinite(bic) and bic < best_bic:
            best_bic = bic
            best = (phi, pp, sigma2)
    return best


def estimate_ar_series(
    y: NDArray[np.float64],
    p_max: int,
    p: object = "auto",
    starts0: Optional[NDArray[Any]] = None,
    center_id: Optional[NDArray[Any]] = None,
) -> dict[str, Any]:
    """Order selection / fixed-p fit for one pooling unit with segments.

    ``y`` is one series or an ``(n, V)`` block of voxels; a block's per-voxel
    autocovariances are pooled exactly as the global/run path pools them
    (fmriAR 5f5fc32), and BIC counts its effective voxels.
    """
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    n = Y.shape[0]
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
    pooled = pooled_acvf_segments(Y, seg_id, p_cap, center_id=center_id)
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

    # Selection on a handful of points would choose AR(8) from 11 frames, so
    # the search (not an explicit p) is bounded by the data.
    p_sel = min(p_cap, n // 5)
    n_bic = _bic_n(Y, n, groups=center_id, vbar=pooled.num[0])
    best_phi, best_p, best_sigma2 = _select_ar_order(
        fit_order, float(gamma0[0]), p_sel, n_bic
    )
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
    censor = _normalize_censor(censor, n)

    corr_by_run: Optional[list[Optional[NDArray[np.float64]]]] = None
    if design is not None:
        _validate_design_residuals(resid, design)
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
