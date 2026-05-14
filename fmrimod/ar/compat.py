"""Backward-compatibility shims for raw-coefficient whitening helpers."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import List, Optional, Union

from numpy.typing import NDArray

from .plan import (
    WhiteningPlan,
    WhitenResult,
)
from .plan import (
    plan_from_phi as _plan_from_phi,
)
from .plan import (
    whiten_with_phi as _whiten_with_phi,
)


@wraps(_plan_from_phi)
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
    """Deprecated shim for :func:`fmrimod.ar.plan_from_phi`."""
    warnings.warn(
        "fmrimod.ar.compat.plan_from_phi() is deprecated; "
        "use fmrimod.ar.plan_from_phi() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _plan_from_phi(
        phi,
        theta,
        runs=runs,
        parcels=parcels,
        pooling=pooling,
        exact_first=exact_first,
        method=method,
    )


@wraps(_whiten_with_phi)
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
    """Deprecated shim for :func:`fmrimod.ar.whiten_with_phi`."""
    warnings.warn(
        "fmrimod.ar.compat.whiten_with_phi() is deprecated; "
        "use fmrimod.ar.whiten_with_phi() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _whiten_with_phi(
        X,
        Y,
        phi,
        theta,
        runs=runs,
        parcels=parcels,
        pooling=pooling,
        exact_first=exact_first,
    )


__all__ = ["plan_from_phi", "whiten_with_phi"]
