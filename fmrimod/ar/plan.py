"""WhiteningPlan and WhitenResult dataclasses + typed constructors.

A WhiteningPlan captures the estimated AR/ARMA noise model and all
parameters needed to apply whitening.  WhitenResult wraps the
whitened design and data matrices. ``plan_from_phi`` and
``whiten_with_phi`` are typed entry points that build a WhiteningPlan
(or apply whitening in one step) from raw AR/MA coefficients without
going through the full ``fit_noise()`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class WhiteningPlan:
    """Complete specification of a whitening transformation.

    Parameters
    ----------
    phi : list of NDArray or None
        Per-run AR coefficients (global/run pooling).  ``None`` for
        parcel-level plans.
    theta : list of NDArray or None
        Per-run MA coefficients.  ``None`` for parcel-level plans.
    order : tuple of (int, int)
        ``(p, q)`` — AR and MA orders.
    runs : NDArray or None
        Integer run labels (length *n*).
    exact_first : bool
        Apply exact AR(1) first-sample scaling at segment starts.
    method : str
        Estimation method: ``"ar"``, ``"arma"``, or ``"afni"``.
    pooling : str
        Pooling mode: ``"global"``, ``"run"``, or ``"parcel"``.
    parcels : NDArray or None
        Voxel-to-parcel mapping (length *V*).
    parcel_ids : list of str or None
        Unique parcel identifiers.
    phi_by_parcel : dict mapping str -> NDArray or None
        Per-parcel AR coefficients.
    theta_by_parcel : dict mapping str -> NDArray or None
        Per-parcel MA coefficients.
    censor : NDArray or None
        0-based indices of censored timepoints.
    """

    phi: list[NDArray] | None = None
    theta: list[NDArray] | None = None
    order: tuple[int, int] = (0, 0)
    runs: NDArray | None = None
    exact_first: bool = False
    method: str = "ar"
    pooling: str = "global"
    parcels: NDArray | None = None
    parcel_ids: list[str] | None = None
    phi_by_parcel: dict[str, NDArray] | None = None
    theta_by_parcel: dict[str, NDArray] | None = None
    censor: NDArray | None = None

    def __repr__(self) -> str:
        p, q = self.order
        lines = [
            "WhiteningPlan",
            f"  Method:  {self.method.upper()}",
            f"  Orders:  p={p}, q={q}",
            f"  Pooling: {self.pooling}",
        ]
        if self.runs is not None:
            n_runs = len(np.unique(self.runs))
            lines.append(f"  Runs:    {n_runs}")
        lines.append(
            f"  Exact first-sample scaling: "
            f"{'AR(1)' if self.exact_first else 'none'}"
        )

        def _fmt(v: NDArray | None) -> str:
            if v is None or len(v) == 0:
                return "(none)"
            return ", ".join(f"{x:.3g}" for x in v)

        if self.pooling == "parcel" and self.phi_by_parcel:
            n_parcels = len(self.phi_by_parcel)
            lines.append(f"  Parcel-level coefficients for {n_parcels} parcels")
            shown = list(self.phi_by_parcel.items())[:5]
            for pid, phi_v in shown:
                theta_v = (self.theta_by_parcel or {}).get(pid)
                s = f"    Parcel {pid}: phi = {_fmt(phi_v)}"
                if theta_v is not None and len(theta_v):
                    s += f"; theta = {_fmt(theta_v)}"
                lines.append(s)
            if n_parcels > 5:
                lines.append("    ...")
        elif self.phi is not None:
            labels = (
                ["global"] if self.pooling == "global"
                else [f"run{i}" for i in range(len(self.phi))]
            )
            for i, (lbl, phi_v) in enumerate(zip(labels, self.phi)):
                if i >= 8:
                    lines.append("    ...")
                    break
                theta_v = self.theta[i] if self.theta and i < len(self.theta) else None
                s = f"    {lbl}: phi = {_fmt(phi_v)}"
                if theta_v is not None and len(theta_v):
                    s += f"; theta = {_fmt(theta_v)}"
                lines.append(s)

        return "\n".join(lines)


@dataclass
class WhitenResult:
    """Whitened design and data matrices.

    Parameters
    ----------
    X : NDArray or None
        Whitened design matrix.  ``None`` for parcel plans.
    Y : NDArray
        Whitened data matrix.
    X_by : dict mapping str -> NDArray or None
        Per-parcel whitened design matrices (parcel plans only).
    """

    X: NDArray | None = None
    Y: NDArray | None = None
    X_by: dict[str, NDArray] | None = None


def plan_from_phi(
    phi: NDArray | list[NDArray],
    theta: NDArray | list[NDArray] | None = None,
    *,
    runs: NDArray | None = None,
    parcels: NDArray | None = None,
    pooling: str = "global",
    exact_first: bool = True,
    method: str | None = None,
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
    phi: NDArray | list[NDArray],
    theta: NDArray | list[NDArray] | None = None,
    *,
    runs: NDArray | None = None,
    parcels: NDArray | None = None,
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
