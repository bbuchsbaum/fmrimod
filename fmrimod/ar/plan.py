"""WhiteningPlan and WhitenResult dataclasses.

A WhiteningPlan captures the estimated AR/ARMA noise model and all
parameters needed to apply whitening.  WhitenResult wraps the
whitened design and data matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

    phi: Optional[List[NDArray]] = None
    theta: Optional[List[NDArray]] = None
    order: Tuple[int, int] = (0, 0)
    runs: Optional[NDArray] = None
    exact_first: bool = False
    method: str = "ar"
    pooling: str = "global"
    parcels: Optional[NDArray] = None
    parcel_ids: Optional[List[str]] = None
    phi_by_parcel: Optional[Dict[str, NDArray]] = None
    theta_by_parcel: Optional[Dict[str, NDArray]] = None
    censor: Optional[NDArray] = None

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

        def _fmt(v: Optional[NDArray]) -> str:
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

    X: Optional[NDArray] = None
    Y: Optional[NDArray] = None
    X_by: Optional[Dict[str, NDArray]] = None
