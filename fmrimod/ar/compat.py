"""Backward-compatibility helpers.

Provides convenience functions for creating whitening plans from raw
AR/MA coefficients without going through the full ``fit_noise()``
pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .plan import WhiteningPlan, WhitenResult


def plan_from_phi(
    phi: Union[NDArray, List[NDArray]],
    theta: Union[NDArray, List[NDArray], None] = None,
    *,
    runs: Optional[NDArray] = None,
    parcels: Optional[NDArray] = None,
    pooling: str = "global",
    exact_first: bool = True,
    method: Optional[str] = None,
) -> WhiteningPlan:
    """Create a WhiteningPlan from raw AR/MA coefficients.

    Parameters
    ----------
    phi : NDArray or list of NDArray
        AR coefficients.  A single array for global pooling, a list for
        run/parcel pooling.
    theta : NDArray or list of NDArray, optional
        MA coefficients.
    runs : NDArray, optional
        Run labels.
    parcels : NDArray, optional
        Parcel labels (required for ``pooling="parcel"``).
    pooling : str
        ``"global"``, ``"run"``, or ``"parcel"``.
    exact_first : bool
        Apply exact first-sample scaling.
    method : str, optional
        Estimation method label.  Auto-detected from theta if not given.

    Returns
    -------
    WhiteningPlan
    """
    if pooling not in ("global", "run", "parcel"):
        raise ValueError("pooling must be 'global', 'run', or 'parcel'")

    # Detect method
    if method is None:
        has_theta = False
        if theta is not None:
            if isinstance(theta, list):
                has_theta = any(len(t) > 0 for t in theta)
            else:
                has_theta = len(np.asarray(theta).ravel()) > 0
        method = "arma" if has_theta else "ar"

    if pooling in ("global", "run"):
        phi_list = phi if isinstance(phi, list) else [np.asarray(phi, dtype=np.float64)]
        if theta is None:
            theta_list = [np.array([], dtype=np.float64)]
        elif isinstance(theta, list):
            theta_list = theta
        else:
            theta_list = [np.asarray(theta, dtype=np.float64)]

        order_p = max(len(p) for p in phi_list)
        order_q = max(len(t) for t in theta_list)

        return WhiteningPlan(
            phi=phi_list,
            theta=theta_list,
            order=(order_p, order_q),
            runs=runs,
            exact_first=exact_first,
            method=method,
            pooling=pooling,
        )

    # Parcel pooling
    if parcels is None:
        raise ValueError("parcels required for pooling='parcel'")
    if not isinstance(phi, dict):
        raise ValueError("phi must be a dict for parcel pooling")
    parcels = np.asarray(parcels, dtype=np.intp)
    phi_by = {str(k): np.asarray(v, dtype=np.float64) for k, v in phi.items()}

    if theta is None:
        theta_by = {k: np.array([], dtype=np.float64) for k in phi_by}
    elif isinstance(theta, dict):
        theta_by = {str(k): np.asarray(v, dtype=np.float64) for k, v in theta.items()}
    else:
        theta_by = {k: np.array([], dtype=np.float64) for k in phi_by}

    order_p = max((len(v) for v in phi_by.values()), default=0)
    order_q = max((len(v) for v in theta_by.values()), default=0)

    return WhiteningPlan(
        phi=None,
        theta=None,
        order=(order_p, order_q),
        runs=runs,
        exact_first=exact_first,
        method=method,
        pooling="parcel",
        parcels=parcels,
        parcel_ids=sorted(phi_by.keys(), key=lambda x: int(x)),
        phi_by_parcel=phi_by,
        theta_by_parcel=theta_by,
    )


def whiten_with_phi(
    X: NDArray,
    Y: NDArray,
    phi: Union[NDArray, List[NDArray]],
    theta: Union[NDArray, List[NDArray], None] = None,
    *,
    runs: Optional[NDArray] = None,
    parcels: Optional[NDArray] = None,
    pooling: str = "global",
    exact_first: bool = False,
) -> WhitenResult:
    """One-step whitening from raw AR/MA coefficients.

    Parameters
    ----------
    X, Y : NDArray
        Design and data matrices.
    phi, theta : NDArray or list
        AR/MA coefficients.
    runs, parcels : NDArray, optional
        Run/parcel labels.
    pooling : str
        Pooling mode.
    exact_first : bool
        Exact first-sample scaling.

    Returns
    -------
    WhitenResult
    """
    from .whitening import whiten_apply

    plan = plan_from_phi(
        phi, theta,
        runs=runs, parcels=parcels,
        pooling=pooling, exact_first=exact_first,
    )
    return whiten_apply(plan, X, Y, runs=runs)
