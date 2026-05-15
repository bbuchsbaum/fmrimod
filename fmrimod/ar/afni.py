"""AFNI restricted AR(3)/AR(5) parameterisation.

Ports ``afni_restricted.R`` — AFNI-style parameterisation of AR polynomials
via a real damping factor and complex conjugate pole pairs.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from .plan import WhiteningPlan


def afni_phi_ar3(a: float, r1: float, t1: float) -> NDArray:
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
) -> NDArray:
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


def afni_restricted_plan(
    resid: NDArray,
    *,
    p: int = 3,
    roots: object = None,
    runs: Optional[NDArray] = None,
    parcels: Optional[NDArray] = None,
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
        Or a dict keyed by parcel id for per-parcel specs.
    runs : NDArray, optional
        Run labels.
    parcels : NDArray, optional
        Parcel labels (triggers parcel-level plan).
    estimate_ma1 : bool
        Estimate MA(1) on AR-filtered residuals.
    exact_first : bool
        Apply exact first-sample AR(1) scaling.

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
        if p == 3:
            return afni_phi_ar3(spec["a"], spec["r1"], spec["t1"])
        else:
            return afni_phi_ar5(spec["a"], spec["r1"], spec["t1"],
                                spec["r2"], spec["t2"])

    if parcels is None:
        # Global plan
        if not isinstance(roots, dict) or "a" not in roots:
            raise ValueError("roots must contain 'a', 'r1', 't1' (and 'r2', 't2' for p=5)")
        phi = _as_phi(cast("dict[str, Any]", roots))
        theta = np.array([], dtype=np.float64)

        if estimate_ma1:
            from .hr_arma import _arma_innovations, hr_arma
            ymean = resid.mean(axis=1)
            ar_resid = _arma_innovations(ymean, phi, np.array([]))
            try:
                est = hr_arma(ar_resid, p=0, q=1, n_iter=0)
                theta = est["theta"]
            except Exception:
                theta = np.array([], dtype=np.float64)

        order_q = len(theta)
        return WhiteningPlan(
            phi=[phi],
            theta=[theta],
            order=(p, order_q),
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

        if estimate_ma1:
            from .hr_arma import _arma_innovations, hr_arma
            cols = np.where(parcels == pid)[0]
            ymean = resid[:, cols].mean(axis=1) if len(cols) > 1 else resid[:, cols[0]]
            ar_resid = _arma_innovations(ymean, phi, np.array([]))
            try:
                est = hr_arma(ar_resid, p=0, q=1, n_iter=0)
                theta_by[key] = est["theta"]
            except Exception:
                theta_by[key] = np.array([], dtype=np.float64)
        else:
            theta_by[key] = np.array([], dtype=np.float64)

    order_q = 1 if estimate_ma1 else 0

    return WhiteningPlan(
        phi=None,
        theta=None,
        order=(p, order_q),
        runs=runs,
        exact_first=exact_first,
        method="afni",
        pooling="parcel",
        parcels=parcels,
        parcel_ids=[str(pid) for pid in pids],
        phi_by_parcel=phi_by,
        theta_by_parcel=theta_by,
    )
