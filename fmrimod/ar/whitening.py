"""AR whitening transforms.

Applies AR pre-whitening to design and data matrices so that
subsequent OLS is equivalent to GLS under the AR noise model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.signal import lfilter

from ._arma_c_backend import arma_whiten_segments_c

try:
    from numba import njit  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional accelerator
    njit = None
    _HAS_NUMBA = False
_USE_NUMBA_ARMA = (
    _HAS_NUMBA and os.environ.get("FMRIMOD_DISABLE_NUMBA_ARMA", "0") != "1"
)
_USE_C_ARMA = os.environ.get("FMRIMOD_DISABLE_C_ARMA", "0") != "1"

from .plan import WhiteningPlan, WhitenResult


def ar_whiten(
    x: NDArray[np.float64],
    phi: NDArray[np.float64],
    *,
    exact_first_ar1: bool = False,
) -> NDArray[np.float64]:
    """Whiten a 1-D or 2-D array using AR coefficients.

    Applies the filter ``(1 - phi_1 L - phi_2 L^2 - ...)`` to remove
    serial correlation.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array.  Shape ``(n,)`` or ``(n, V)``.
    phi : NDArray[np.float64]
        AR coefficients, shape ``(p,)``.
    exact_first_ar1 : bool
        If ``True`` and *phi* is AR(1), scale the first sample by
        ``sqrt(1 - phi[0] ** 2)``. The default ``False`` matches the
        low-level fmriAR conditional whitening convention.

    Returns
    -------
    NDArray[np.float64]
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

    for k in range(1, p + 1):
        if n > k:
            result[k:, :] -= phi[k - 1] * x[:-k, :]

    if exact_first_ar1 and p == 1 and n:
        scale = np.sqrt(max(1.0 - phi[0] ** 2, 0.0))
        result[0] = x[0] * scale

    if was_1d:
        return result.ravel()
    return result


def ar_whiten_matrix(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    *,
    exact_first_ar1: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Whiten both design and data matrices with AR coefficients.

    Parameters
    ----------
    X : NDArray[np.float64]
        Design matrix, shape ``(n, p)``.
    Y : NDArray[np.float64]
        Data matrix, shape ``(n, V)``.
    phi : NDArray[np.float64]
        AR coefficients.  Shape ``(ar_order,)`` for global or
        ``(ar_order, V)`` for voxelwise.
    exact_first_ar1 : bool
        If ``True`` and AR order is 1, scale the first whitened sample by
        ``sqrt(1 - phi[0] ** 2)``. Defaults to ``False`` to match fmriAR's
        low-level whitening default.

    Returns
    -------
    X_w, Y_w : tuple of NDArray[np.float64]
        Whitened design and data matrices.
    """
    phi = np.asarray(phi, dtype=np.float64)

    if phi.ndim == 1:
        # Global phi: apply same whitening to X and Y
        return (
            ar_whiten(X, phi, exact_first_ar1=exact_first_ar1),
            ar_whiten(Y, phi, exact_first_ar1=exact_first_ar1),
        )
    else:
        # Voxelwise phi: X gets whitened with mean phi,
        # Y gets whitened voxel-by-voxel
        phi_mean = np.mean(phi, axis=1)
        X_w = ar_whiten(X, phi_mean, exact_first_ar1=exact_first_ar1)

        Y_w = Y.copy()
        ar_order = phi.shape[0]
        n, V = Y.shape
        for v in range(V):
            phi_v = phi[:, v]
            for k in range(1, ar_order + 1):
                if n > k:
                    Y_w[k:, v] -= phi_v[k - 1] * Y[:-k, v]
            if exact_first_ar1 and ar_order == 1 and n:
                scale = np.sqrt(max(1.0 - phi_v[0] ** 2, 0.0))
                Y_w[0, v] = Y[0, v] * scale

        return X_w, Y_w


def ar_covariance_matrix(
    phi: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Build the AR(p) covariance matrix (Toeplitz).

    Parameters
    ----------
    phi : NDArray[np.float64]
        AR coefficients, shape ``(p,)``.
    n : int
        Matrix dimension (number of timepoints).

    Returns
    -------
    NDArray[np.float64]
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
    return cast("NDArray[np.float64]", linalg.toeplitz(r))


# ---------------------------------------------------------------------------
# Segment-aware ARMA whitening (ports fmriAR_whiten.cpp)
# ---------------------------------------------------------------------------


def _arma_whiten_segments_numba_core(
    y: NDArray[np.float64],
    phi: NDArray[np.float64],
    theta: NDArray[np.float64],
    seg_starts: NDArray[np.float64],
    do_exact: bool,
) -> NDArray[np.float64]:
    """Placeholder numba backend symbol for monkeypatching in tests."""
    raise RuntimeError("Numba ARMA backend unavailable")


if _USE_NUMBA_ARMA:

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _arma_whiten_segments_numba_core(
        y: NDArray[np.float64],
        phi: NDArray[np.float64],
        theta: NDArray[np.float64],
        seg_starts: NDArray[np.float64],
        do_exact: bool,
    ) -> NDArray[np.float64]:
        """Numba core for segment-aware ARMA whitening."""
        n, v = y.shape
        p = phi.shape[0]
        q = theta.shape[0]
        out = np.empty_like(y)

        n_seg = seg_starts.shape[0]
        for si in range(n_seg):
            s_start = int(seg_starts[si])
            s_end = int(seg_starts[si + 1]) if (si + 1) < n_seg else n
            if s_end <= s_start:
                continue

            for col in range(v):
                for t in range(s_start, s_end):
                    val = y[t, col]

                    # AR contribution uses original y.
                    for k in range(p):
                        tt = t - (k + 1)
                        if tt >= s_start:
                            val -= phi[k] * y[tt, col]

                    # MA contribution uses previous innovations (out).
                    for j in range(q):
                        tt = t - (j + 1)
                        if tt >= s_start:
                            val -= theta[j] * out[tt, col]

                    out[t, col] = val

                if do_exact:
                    s = 1.0 - phi[0] * phi[0]
                    if s < 0.0:
                        s = 0.0
                    out[s_start, col] *= np.sqrt(s)

        return out


# Rows of the exact start-up factor this close to the stationary limit
# (1, theta_1, ..., theta_q, 0, ...) are treated as converged.
_EXACT_FACTOR_TOL = 1e-14


def _arma_acvf_unit(
    phi: NDArray[np.float64], theta: NDArray[np.float64], max_lag: int
) -> NDArray[np.float64] | None:
    """Unit-innovation ARMA autocovariance at lags ``0..max_lag``.

    Solves the exact linear system of Brockwell & Davis (1991, sec. 3.3), as
    fmriAR_whiten.cpp ``arma_acvf_unit`` does. ``None`` when the system is
    singular or ``gamma_0 <= 0``, i.e. the AR part is not stationary.
    """
    p, q = phi.size, theta.size
    th = np.concatenate([[1.0], theta])
    psi = np.zeros(q + 1)
    psi[0] = 1.0
    for j in range(1, q + 1):
        psi[j] = th[j] + sum(phi[k - 1] * psi[j - k] for k in range(1, min(j, p) + 1))
    rhs = np.array(
        [float(np.dot(th[k:], psi[: q + 1 - k])) if k <= q else 0.0
         for k in range(max(max_lag, p) + 1)]
    )
    A = np.eye(p + 1)
    for k in range(p + 1):
        for j in range(1, p + 1):
            A[k, abs(k - j)] -= phi[j - 1]
    try:
        g = np.linalg.solve(A, rhs[: p + 1])
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(g)) or not g[0] > 0:
        return None
    gamma = np.zeros(rhs.size)
    gamma[: p + 1] = g
    for k in range(p + 1, rhs.size):
        gamma[k] = rhs[k] + float(np.dot(phi, gamma[k - 1 :: -1][:p]))
    return gamma[: max_lag + 1]


def _exact_factor(
    phi: NDArray[np.float64], theta: NDArray[np.float64], n: int
) -> NDArray[np.float64] | None:
    """Banded Cholesky factor for exact stationary start-up (Ansley 1979).

    With ``m = max(p, q)``, ``z_t = y_t`` for ``t < m`` and ``phi(B) y_t``
    afterwards. ``Cov(z)`` is banded with half-bandwidth ``m`` and ``z = A y``
    with ``A`` unit lower triangular, so its Cholesky factor ``L`` satisfies
    ``L^-1 z = chol(Sigma)^-1 y``: exact GLS whitening of a stationary segment.
    Row ``t`` holds ``L[t, t - d]`` at column ``d``. A shorter segment's factor
    is the leading block of a longer one's, so one factor serves every
    segment. Rows stop once they reach the stationary limit
    ``(1, theta_1, ..., theta_q, 0, ...)``; later rows follow it. ``None``
    when the AR part is not stationary (fmriAR_whiten.cpp
    ``build_exact_factor``).
    """
    p, q = phi.size, theta.size
    m = max(p, q)
    gamma = _arma_acvf_unit(phi, theta, m + p)
    if gamma is None:
        return None
    th = np.concatenate([[1.0], theta])
    # Cov(z_i, z_j), i <= j, by band offset d = j - i: both in the start-up
    # block, the start-up row against a filtered one, and two filtered rows.
    cross = np.array(
        [gamma[d] - sum(phi[k - 1] * gamma[abs(d - k)] for k in range(1, p + 1))
         for d in range(m + 1)]
    )
    ma_cov = np.array(
        [float(np.dot(th[: q + 1 - d], th[d:])) if d <= q else 0.0
         for d in range(m + 1)]
    )
    limit = np.zeros(m + 1)
    limit[0] = 1.0
    limit[1 : q + 1] = theta
    tol = _EXACT_FACTOR_TOL * max(1.0, float(np.max(np.abs(limit))))
    F = np.zeros((n, m + 1))
    for t in range(n):
        j0 = max(0, t - m)
        for j in range(j0, t + 1):
            d = t - j
            if t < m:
                s = gamma[d]
            elif j < m:
                s = cross[d]
            else:
                s = ma_cov[d]
            for k in range(max(j0, j - m), j):
                s -= F[t, t - k] * F[j, j - k]
            if j == t:
                if not (s > 0 and np.isfinite(s)):
                    return None
                F[t, 0] = np.sqrt(s)
            else:
                F[t, d] = s / F[j, 0]
        if t >= m and np.max(np.abs(F[t] - limit)) <= tol:
            return F[:t]
    return F


def _filter_exact(
    seg: NDArray[np.float64],
    phi: NDArray[np.float64],
    theta: NDArray[np.float64],
    factor: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact stationary-start whitening of one segment, all columns at once."""
    n_seg = seg.shape[0]
    p, q = phi.size, theta.size
    m = factor.shape[1] - 1
    z = seg.copy()
    if n_seg > m:
        for k in range(1, p + 1):
            z[m:] -= phi[k - 1] * seg[m - k : n_seg - k]
    out = np.empty_like(seg)
    n_rows = min(n_seg, factor.shape[0])
    for t in range(n_rows):
        acc = z[t].copy()
        for d in range(1, min(t, m) + 1):
            acc -= factor[t, d] * out[t - d]
        out[t] = acc / factor[t, 0]
    # Converged rows follow the limit: out_t = z_t - sum_d theta_d out_{t-d},
    # the conditional MA recursion seeded with the exact outputs so far.
    if q == 0:
        out[n_rows:] = z[n_rows:]
    else:
        for t in range(n_rows, n_seg):
            acc = z[t].copy()
            for d in range(1, q + 1):
                acc -= theta[d - 1] * out[t - d]
            out[t] = acc
    return out


def arma_whiten_segments(
    y: NDArray[np.float64],
    phi: NDArray[np.float64],
    theta: NDArray[np.float64],
    seg_starts: NDArray[Any],
    exact_first_ar1: bool = False,
) -> NDArray[np.float64]:
    """Apply ARMA whitening filter with segment resets.

    At each segment boundary the filter state is reset so that
    cross-run / cross-censor-gap contamination is avoided.

    Parameters
    ----------
    y : NDArray[np.float64]
        Input array, shape ``(n,)`` or ``(n, V)``.
    phi : NDArray[np.float64]
        AR coefficients, shape ``(p,)``.
    theta : NDArray[np.float64]
        MA coefficients, shape ``(q,)``.
    seg_starts : NDArray[np.float64]
        0-based segment start indices (must include 0).
    exact_first_ar1 : bool
        If ``True``, start every segment with the exact stationary
        initialisation of the AR/ARMA model (fmriAR 617bd87), so the
        result is exact GLS whitening within each segment. For AR(1) this
        is the Prais-Winsten ``sqrt(1 - phi[0]**2)`` scaling of the first
        sample. A non-stationary AR part falls back to the conditional
        filter. The name is historical.

    Returns
    -------
    NDArray[np.float64]
        Whitened array, same shape as *y*.
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)

    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, np.newaxis]

    n, _ = y.shape
    p = len(phi)
    q_ord = len(theta)

    seg_starts = np.sort(np.unique(np.asarray(seg_starts, dtype=np.intp)))
    seg_ends = np.append(seg_starts[1:], n)

    out = np.empty_like(y)

    if exact_first_ar1 and (p > 0 or q_ord > 0) and n > 0:
        max_len = int(np.max(seg_ends - seg_starts))
        factor = _exact_factor(phi, theta, max_len)
        if factor is not None:
            for s_start, s_end in zip(seg_starts, seg_ends, strict=True):
                if s_end > s_start:
                    out[s_start:s_end] = _filter_exact(
                        y[s_start:s_end], phi, theta, factor
                    )
            return out.ravel() if was_1d else out
        # Non-stationary AR part: there is no stationary start-up, so use the
        # conditional recursion (fmriAR_whiten.cpp:248-254).

    # Build filter coefficients: b = [1, -phi_1, ...], a = [1, theta_1, ...]
    b = np.concatenate([[1.0], -phi]) if p > 0 else np.array([1.0])
    a = np.concatenate([[1.0], theta]) if q_ord > 0 else np.array([1.0])

    do_exact = False

    # ARMA path: prefer numba when available. On the benchmark gate workload,
    # numba avoids ctypes boundary overhead and is faster than the C shim.
    if q_ord > 0 and _USE_NUMBA_ARMA:
        out = _arma_whiten_segments_numba_core(
            np.ascontiguousarray(y),
            np.ascontiguousarray(phi),
            np.ascontiguousarray(theta),
            np.ascontiguousarray(seg_starts),
            do_exact,
        )
        if was_1d:
            return out.ravel()
        return out

    if q_ord > 0 and _USE_C_ARMA:
        out_c = arma_whiten_segments_c(
            y=np.ascontiguousarray(y),
            phi=np.ascontiguousarray(phi),
            theta=np.ascontiguousarray(theta),
            seg_starts=np.ascontiguousarray(seg_starts),
            do_exact=do_exact,
        )
        if out_c is not None:
            if was_1d:
                return out_c.ravel()
            return out_c

    for s_start, s_end in zip(seg_starts, seg_ends):
        seg = y[s_start:s_end]
        seg_len = seg.shape[0]
        if seg_len == 0:
            continue

        # Pure AR case (q=0): apply FIR filter directly with vectorized
        # shifted subtraction, avoiding scipy lfilter overhead.
        if q_ord == 0 and p > 0:
            seg_out = seg.copy()
            for k in range(1, p + 1):
                if seg_len > k:
                    seg_out[k:, :] -= phi[k - 1] * seg[:-k, :]
            out[s_start:s_end, :] = seg_out
        else:
            # General ARMA case: use scipy lfilter along time axis.
            out[s_start:s_end, :] = lfilter(b, a, seg, axis=0)

    if was_1d:
        return out.ravel()
    return out


def _sub_run_starts(n_run: int, censor_rel: NDArray[Any]) -> NDArray[np.intp]:
    """Compute 0-based sub-run starts from relative censor indices."""
    starts = [0]
    if len(censor_rel) > 0:
        add = censor_rel + 1
        add = add[add < n_run]
        starts = sorted(set(starts) | set(add.tolist()))
    return np.array(starts, dtype=np.intp)


def _full_run_starts(
    runs: NDArray[Any], censor: Optional[NDArray[Any]], n: int
) -> NDArray[np.intp]:
    """Merge run boundaries and censor gaps into segment starts."""
    from .acvf import run_codes

    starts = {0}
    if runs is not None:
        codes = run_codes(runs, n)
        diffs = np.where(np.diff(codes) != 0)[0] + 1
        starts.update(int(x) for x in diffs.tolist())
    if censor is not None and len(censor) > 0:
        censor = np.asarray(censor, dtype=np.intp)
        extra = censor + 1
        extra = extra[extra < n]
        starts.update(int(x) for x in extra.tolist())
    return np.array(sorted(starts), dtype=np.intp)


def whiten_apply(
    plan: "WhiteningPlan",
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    runs: Optional[NDArray[np.float64]] = None,
    censor: Optional[NDArray[np.float64]] = None,
    parcels: Optional[NDArray[np.float64]] = None,
) -> "WhitenResult":
    """Apply a whitening plan to design and data matrices.

    Parameters
    ----------
    plan : WhiteningPlan
        Plan from :func:`~fmrimod.ar.estimation.fit_noise`.
    X : NDArray[np.float64]
        Design matrix, shape ``(n, k)``.
    Y : NDArray[np.float64]
        Data matrix, shape ``(n, V)``.
    runs : NDArray[np.float64], optional
        Run labels (overrides plan's runs).
    censor : NDArray[np.float64], optional
        0-based censor indices, or a boolean mask with one entry per row
        (overrides plan's censor).
    parcels : NDArray[np.float64], optional
        Parcel labels (overrides plan's parcels).

    Returns
    -------
    WhitenResult
    """
    from .acvf import _normalize_censor, parcel_codes, run_codes, run_sets
    from .plan import WhitenResult

    X = np.array(X, dtype=np.float64, copy=True)
    Y = np.array(Y, dtype=np.float64, copy=True)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("whiten_apply: NA values detected in X or Y")

    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")

    # Resolve runs
    if runs is None and plan.runs is not None and len(plan.runs) == n:
        runs = plan.runs
    if runs is None:
        runs = np.ones(n, dtype=np.intp)
    else:
        run_codes(runs, n)

    # Resolve censor: None inherits the plan's set, an explicit empty one
    # means no censoring; a boolean mask must not be cast to 0/1 indices.
    censor_idx = _normalize_censor(plan.censor if censor is None else censor, n)

    # --- Parcel plan ---
    if plan.pooling == "parcel":
        parcels_vec = parcels if parcels is not None else plan.parcels
        if parcels_vec is None:
            raise ValueError("Parcel labels required for parcel plan")
        parcels_vec = parcel_codes(parcels_vec)

        run_starts_vec = _full_run_starts(runs, censor_idx, n)
        phi_by = plan.phi_by_parcel or {}
        theta_by = plan.theta_by_parcel or {}

        Yw = np.empty_like(Y)
        X_by: Dict[str, NDArray[np.float64]] = {}

        for pid in plan.parcel_ids or sorted(phi_by.keys()):
            key = str(pid)
            cols = np.where(parcels_vec == int(pid))[0]
            if len(cols) == 0:
                continue

            phi_v = phi_by.get(key, np.array([], dtype=np.float64))
            theta_v = theta_by.get(key, np.array([], dtype=np.float64))

            if len(theta_v) == 0:
                XY_sub = np.hstack((X, Y[:, cols]))
                XYw_sub = arma_whiten_segments(
                    XY_sub,
                    phi_v,
                    theta_v,
                    run_starts_vec,
                    exact_first_ar1=plan.exact_first,
                )
                k = X.shape[1]
                X_sub = XYw_sub[:, :k]
                Y_sub = XYw_sub[:, k:]
            else:
                Y_sub = arma_whiten_segments(
                    Y[:, cols],
                    phi_v,
                    theta_v,
                    run_starts_vec,
                    exact_first_ar1=plan.exact_first,
                )
                X_sub = arma_whiten_segments(
                    X,
                    phi_v,
                    theta_v,
                    run_starts_vec,
                    exact_first_ar1=plan.exact_first,
                )
            Yw[:, cols] = Y_sub
            X_by[key] = X_sub

        return WhitenResult(X=None, Y=Yw, X_by=X_by)

    # --- Global / run plan ---
    rsplits = [idx for _, idx in run_sets(runs, n)]

    def _row_selector(idx: NDArray[Any]) -> Any:
        """Prefer contiguous slice selectors to avoid fancy-index copies."""
        if len(idx) == 0:
            return idx
        if len(idx) == 1:
            i0 = int(idx[0])
            return slice(i0, i0 + 1)
        i0 = int(idx[0])
        i1 = int(idx[-1])
        if (i1 - i0 + 1) == len(idx) and np.all(np.diff(idx) == 1):
            return slice(i0, i1 + 1)
        return idx

    # Split censor by run
    censor_by_run = [np.array([], dtype=np.intp) for _ in rsplits]
    if censor_idx is not None:
        for ri, idx in enumerate(rsplits):
            start = idx[0]
            c_in = np.intersect1d(censor_idx, idx)
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

    Xw = np.empty_like(X)
    Yw = np.empty_like(Y)

    for ri, idx in enumerate(rsplits):
        row_sel = _row_selector(idx)
        Xr = X[row_sel]
        Yr = Y[row_sel]
        phi_r = phi_list[ri] if ri < len(phi_list) else np.array([], dtype=np.float64)
        theta_r = (
            theta_list[ri] if ri < len(theta_list) else np.array([], dtype=np.float64)
        )

        seg_starts = _sub_run_starts(len(idx), censor_by_run[ri])

        if len(theta_r) == 0:
            XYr = np.hstack((Xr, Yr))
            XYw_r = arma_whiten_segments(
                XYr, phi_r, theta_r, seg_starts, exact_first_ar1=plan.exact_first
            )
            k = Xr.shape[1]
            Xw_r = XYw_r[:, :k]
            Yw_r = XYw_r[:, k:]
        else:
            Xw_r = arma_whiten_segments(
                Xr, phi_r, theta_r, seg_starts, exact_first_ar1=plan.exact_first
            )
            Yw_r = arma_whiten_segments(
                Yr, phi_r, theta_r, seg_starts, exact_first_ar1=plan.exact_first
            )
        Xw[row_sel, :] = Xw_r
        Yw[row_sel, :] = Yw_r

    return WhitenResult(X=Xw, Y=Yw)


def whiten(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    runs: Optional[NDArray[np.float64]] = None,
    censor: Optional[NDArray[np.float64]] = None,
    **fit_kwargs: object,
) -> "WhitenResult":
    """Fit noise model and apply whitening in one call.

    Convenience function: computes OLS residuals, calls
    :func:`~fmrimod.ar.estimation.fit_noise`, then
    :func:`whiten_apply`.

    Parameters
    ----------
    X, Y : NDArray[np.float64]
        Design and data matrices.
    runs : NDArray[np.float64], optional
        Run labels.
    censor : NDArray[np.float64], optional
        0-based censor indices, or a boolean mask with one entry per row.
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
    plan = fit_noise(
        resid=res, runs=runs, censor=censor, **cast("dict[str, Any]", fit_kwargs)
    )
    return whiten_apply(plan, X, Y, runs=runs, censor=censor)
