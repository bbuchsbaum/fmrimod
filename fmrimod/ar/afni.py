"""AFNI restricted AR(3)/AR(5) parameterisation.

Ports ``afni_restricted.R`` — AFNI-style parameterisation of AR polynomials
via a real damping factor and complex conjugate pole pairs.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from .plan import WhiteningPlan


def afni_phi_ar3(a: float, r1: float, t1: float) -> NDArray[np.float64]:
    """Convert AFNI AR(3) root parameters to AR coefficients.

    Parameters
    ----------
    a : float
        Real damping factor, clipped to [0, 0.95].
    r1 : float
        Radius of complex pole pair, clipped to [0, 0.95].
    t1 : float
        Angle of complex pole pair (radians), clipped to [0, pi].

    Returns
    -------
    NDArray
        AR(3) coefficients, shape ``(3,)``.
    """
    a = np.clip(a, 0.0, 0.95)
    r1 = np.clip(r1, 0.0, 0.95)
    t1 = np.clip(t1, 0.0, np.pi)
    c1 = np.cos(t1)
    p1 = a + 2 * r1 * c1
    p2 = -2 * a * r1 * c1 - r1 ** 2
    p3 = a * r1 ** 2
    return np.array([p1, p2, p3], dtype=np.float64)


def afni_phi_ar5(
    a: float, r1: float, t1: float, r2: float, t2: float
) -> NDArray[np.float64]:
    """Convert AFNI AR(5) root parameters to AR coefficients.

    Parameters
    ----------
    a : float
        Real damping factor, clipped to [0, 0.95].
    r1 : float
        Radius of first complex pole pair, clipped to [0, 0.95].
    t1 : float
        Angle of first complex pole pair (radians), clipped to [0, pi].
    r2 : float
        Radius of second complex pole pair, clipped to [0, 0.95].
    t2 : float
        Angle of second complex pole pair (radians), clipped to [0, pi].

    Returns
    -------
    NDArray
        AR(5) coefficients, shape ``(5,)``.
    """
    a = np.clip(a, 0.0, 0.95)
    r1 = np.clip(r1, 0.0, 0.95)
    r2 = np.clip(r2, 0.0, 0.95)
    t1 = np.clip(t1, 0.0, np.pi)
    t2 = np.clip(t2, 0.0, np.pi)
    c1, c2 = np.cos(t1), np.cos(t2)

    p1 = 2 * r1 * c1 + 2 * r2 * c2 + a
    p2 = -4 * r1 * r2 * c1 * c2 - 2 * a * (r1 * c1 + r2 * c2) - r1 ** 2 - r2 ** 2
    p3 = a * (r1 ** 2 + r2 ** 2 + 4 * r1 * r2 * c1 * c2) + 2 * r1 * r2 * (r2 * c1 + r1 * c2)
    p4 = -2 * a * r1 * r2 * (r2 * c1 + r1 * c2) - (r1 ** 2) * (r2 ** 2)
    p5 = a * (r1 ** 2) * (r2 ** 2)
    return np.array([p1, p2, p3, p4, p5], dtype=np.float64)


def _check_afni_spec(spec: object, p: int) -> None:
    """Validate one AFNI root spec as AFNI (``armacor.c``) does.

    Negative pole parameters are rejected rather than clamped: clamping
    turned a sign error into a different noise model (fmriAR 5f5fc32).
    Values above 0.95 and angles outside ``[0, pi]`` are still clamped by
    :func:`afni_phi_ar3` / :func:`afni_phi_ar5`, as in AFNI.
    """
    if not isinstance(spec, dict):
        raise ValueError("an AFNI root spec must be a dict of named parameters")
    need = ("a", "r1", "t1") if p == 3 else ("a", "r1", "t1", "r2", "t2")
    miss = [k for k in need if k not in spec]
    if miss:
        raise ValueError(f"AFNI root spec is missing: {', '.join(miss)}")
    try:
        vals = np.array([spec[k] for k in need], dtype=np.float64)
    except (TypeError, ValueError):
        vals = np.array([np.nan])
    if vals.shape != (len(need),) or not np.all(np.isfinite(vals)):
        raise ValueError("AFNI root parameters must be finite numbers")
    mods = [float(spec[k]) for k in ("a", "r1", "r2") if k in need]
    if min(mods) < 0:
        raise ValueError(
            "AFNI root parameters a, r1, r2 must be >= 0 (AFNI rejects "
            "negative values)"
        )
    vrt = spec.get("vrt", 1.0)
    try:
        vrt_arr = np.asarray(vrt, dtype=np.float64)
    except (TypeError, ValueError):
        vrt_arr = np.array([np.nan])
    if vrt_arr.size != 1 or not np.isfinite(vrt_arr).all() or float(vrt_arr) <= 0:
        raise ValueError("'vrt' must be a single number in (0, 1]")


def _afni_ma1_pooled(
    resid_cols: NDArray[np.float64],
    phi: NDArray[np.float64],
    run_starts0: NDArray[np.intp],
) -> NDArray[np.float64] | None:
    """MA(1) for AFNI's additive white noise, from the AR-filtered voxels.

    Pooled Hannan-Rissanen over the voxels of every run (fmriAR 617bd87,
    5f5fc32 ``.afni_ma1_pooled``). The filtered voxel-mean series it
    replaces measures what the voxels share rather than the voxel noise
    being whitened. The first ``p`` frames of each run have no AR
    pre-sample and are dropped; runs with fewer than 10 frames left are
    skipped. ``None`` when no estimate can be made.
    """
    import warnings

    from .hr_arma import _hr_arma_pooled
    from .whitening import arma_whiten_segments

    n = resid_cols.shape[0]
    filt = arma_whiten_segments(resid_cols, phi, np.array([]), run_starts0)
    ends = np.append(run_starts0[1:], n)
    units = [
        (filt[s + phi.size : e], np.array([0], dtype=np.intp))
        for s, e in zip(run_starts0, ends, strict=True)
        if e - (s + phi.size) >= 10
    ]
    if not units:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = _hr_arma_pooled(units, 0, 1)
    return est.theta if est.ok else None


def afni_restricted_plan(
    resid: NDArray[np.float64],
    *,
    p: int = 3,
    roots: object = None,
    runs: Optional[NDArray[Any]] = None,
    parcels: Optional[NDArray[Any]] = None,
    estimate_ma1: bool = True,
    exact_first: bool = True,
) -> WhiteningPlan:
    """Build an AFNI-style restricted AR plan from root parameters.

    Parameters
    ----------
    resid : NDArray
        Residual matrix, shape ``(n, V)``.
    p : int
        AR order, must be 3 or 5.
    roots : dict or dict-of-dicts
        For ``p=3``: ``{"a", "r1", "t1"}``.
        For ``p=5``: ``{"a", "r1", "t1", "r2", "t2"}``.
        Or a dict keyed by parcel id for per-parcel specs. As in AFNI,
        ``a``, ``r1``, ``r2`` must be non-negative (values above 0.95 are
        clamped) and angles are clamped to ``[0, pi]``; every parameter
        must be finite. An optional ``vrt`` must lie in ``(0, 1]``.
    runs : NDArray, optional
        Run labels.
    parcels : NDArray, optional
        Parcel labels (triggers parcel-level plan).
    estimate_ma1 : bool
        Estimate an MA(1) term, standing in for AFNI's additive white
        noise, by Hannan-Rissanen pooled over the AR-filtered voxels of
        each run (per parcel for parcel plans).
    exact_first : bool
        Exact stationary start-up at each segment start.

    Returns
    -------
    WhiteningPlan
    """
    if p not in (3, 5):
        raise ValueError(f"p must be 3 or 5, got {p}")

    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim == 1:
        resid = resid[:, np.newaxis]
    n, v = resid.shape

    def _as_phi(spec: dict[str, Any]) -> NDArray[np.float64]:
        _check_afni_spec(spec, p)
        if p == 3:
            return afni_phi_ar3(spec["a"], spec["r1"], spec["t1"])
        else:
            return afni_phi_ar5(spec["a"], spec["r1"], spec["t1"],
                                spec["r2"], spec["t2"])

    if runs is None:
        run_starts0 = np.array([0], dtype=np.intp)
    else:
        from .acvf import run_codes

        codes = run_codes(runs, n)
        run_starts0 = np.flatnonzero(np.diff(codes, prepend=codes[0] - 1))

    if parcels is None:
        # Global plan
        if not isinstance(roots, dict):
            raise ValueError("roots must contain 'a', 'r1', 't1' (and 'r2', 't2' for p=5)")
        phi = _as_phi(cast("dict[str, Any]", roots))
        theta = np.array([], dtype=np.float64)

        if estimate_ma1:
            th = _afni_ma1_pooled(resid, phi, run_starts0)
            if th is not None:
                theta = th

        return WhiteningPlan(
            phi=[phi],
            theta=[theta],
            order=(len(phi), len(theta)),
            runs=runs,
            exact_first=exact_first,
            method="afni",
            pooling="global" if runs is None else "run",
        )

    # Parcel plan
    parcels = np.asarray(parcels, dtype=np.intp)
    pids = np.unique(parcels)
    phi_by = {}
    theta_by = {}

    # Determine if roots is a single spec or per-parcel
    is_single = isinstance(roots, dict) and "a" in roots

    roots_dict = cast("dict[Any, Any]", roots)
    for pid in pids:
        key = str(pid)
        spec = roots_dict if is_single else roots_dict.get(key, roots_dict.get(pid))
        if spec is None:
            # Fall back to first available
            spec = roots_dict if is_single else next(iter(roots_dict.values()))
        phi = _as_phi(cast("dict[str, Any]", spec))
        phi_by[key] = phi

        theta_by[key] = np.array([], dtype=np.float64)
        if estimate_ma1:
            th = _afni_ma1_pooled(resid[:, parcels == pid], phi, run_starts0)
            if th is not None:
                theta_by[key] = th

    # The order the plan actually carries: an MA(1) estimate can fail.
    order_p = max((len(v) for v in phi_by.values()), default=0)
    order_q = max((len(v) for v in theta_by.values()), default=0)

    return WhiteningPlan(
        phi=None,
        theta=None,
        order=(order_p, order_q),
        runs=runs,
        exact_first=exact_first,
        method="afni",
        pooling="parcel",
        parcels=parcels,
        parcel_ids=[str(pid) for pid in pids],
        phi_by_parcel=phi_by,
        theta_by_parcel=theta_by,
    )
