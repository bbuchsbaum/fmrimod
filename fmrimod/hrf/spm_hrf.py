"""SPM HRF implementations with analytic derivatives.

Typed dataclass subclasses of :class:`HRF`. Each SPMG kind exposes its
double-gamma parameters as named fields (``p1``, ``p2``, ``a1``). The
inherited ``params`` / ``param_names`` mirror is kept as a transition
surface; evaluation reads the typed fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from .derivatives import (
    spmg1_derivative,
    spmg1_dispersion_derivative,
    spmg1_second_derivative,
)
from .functions import spm_canonical

_SPM_PARAM_NAMES = ("p1", "p2", "a1")


def _spm_params(p1: float, p2: float, a1: float) -> dict[str, float]:
    return {"p1": p1, "p2": p2, "a1": a1}


@dataclass
class SPMG1_HRF(HRF):
    """SPM canonical HRF with analytic derivative support."""

    name: str = "SPMG1"
    nbasis: int = 1
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __post_init__(self) -> None:
        self.params = _spm_params(self.p1, self.p2, self.a1)
        self.param_names = list(_SPM_PARAM_NAMES)

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return spm_canonical(t, p1=self.p1, p2=self.p2, a1=self.a1)

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        return spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)


@dataclass
class SPMG2_HRF(HRF):
    """SPM HRF with temporal derivative basis."""

    name: str = "SPMG2"
    nbasis: int = 2
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __post_init__(self) -> None:
        self.params = _spm_params(self.p1, self.p2, self.a1)
        self.param_names = list(_SPM_PARAM_NAMES)

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical(t, p1=self.p1, p2=self.p2, a1=self.a1)
        derivative = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        result = np.column_stack([canonical, derivative])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        first_deriv = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        second_deriv = spmg1_second_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        result = np.column_stack([first_deriv, second_deriv])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result


@dataclass
class SPMG3_HRF(HRF):
    """SPM informed basis: canonical + temporal + dispersion derivatives.

    The third column is the **SPM dispersion derivative** — the
    finite-difference partial derivative of the canonical with
    respect to its dispersion (width) parameter, taken at
    ``σ = 1`` with step ``dx = 0.01`` and SPM's sign convention
    ``(h(σ=1) - h(σ=1+dx)) / dx``. This matches the SPM informed
    basis (``spm_get_bf``: "hrf (with time and dispersion
    derivatives)") and Nilearn's
    ``spm_dispersion_derivative``.

    The third column is **not** the second time derivative
    ``∂²h/∂t²``. Earlier releases (matching R ``fmrireg::HRF_SPMG3``)
    used the second time derivative; that was a divergence from the
    SPM definition and from the interpretive claim that the third
    coefficient captures HRF *width* changes. The second-time-
    derivative basis remains available via
    :func:`spmg1_second_derivative` for callers that need the
    legacy R-compatible shape.
    """

    name: str = "SPMG3"
    nbasis: int = 3
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __post_init__(self) -> None:
        self.params = _spm_params(self.p1, self.p2, self.a1)
        self.param_names = list(_SPM_PARAM_NAMES)

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical(t, p1=self.p1, p2=self.p2, a1=self.a1)
        derivative = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        dispersion = spmg1_dispersion_derivative(
            t, p1=self.p1, p2=self.p2, a1=self.a1
        )
        result = np.column_stack([canonical, derivative, dispersion])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result
