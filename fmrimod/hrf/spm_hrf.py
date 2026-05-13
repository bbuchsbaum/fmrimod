"""SPM HRF implementations with analytic derivatives.

Typed dataclass subclasses of :class:`HRF`. Each SPMG kind exposes its
double-gamma parameters as named fields (``p1``, ``p2``, ``a1``) instead
of stuffing them into the inherited ``params`` dict. The ``params``
mirror is preserved during the transition window for cross_testing
readers; see bead ``bd-01KRGCZJ6JAA4BKRTNQ91P2PE5`` for its retirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from .derivatives import spmg1_derivative, spmg1_second_derivative
from .functions import spm_canonical

_SPM_PARAM_NAMES: tuple[str, ...] = ("p1", "p2", "a1")


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
        self.params = {"p1": self.p1, "p2": self.p2, "a1": self.a1}
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
        self.params = {"p1": self.p1, "p2": self.p2, "a1": self.a1}
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
    """SPM HRF with temporal and dispersion derivatives."""

    name: str = "SPMG3"
    nbasis: int = 3
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    # Step size for the numerical derivative of the dispersion (second-derivative)
    # basis used inside ``_derivative``. Kept as a class-level constant so the
    # value is auditable rather than hidden behind a magic literal.
    _DISPERSION_DERIV_DX: ClassVar[float] = 1e-3

    def __post_init__(self) -> None:
        self.params = {"p1": self.p1, "p2": self.p2, "a1": self.a1}
        self.param_names = list(_SPM_PARAM_NAMES)

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical(t, p1=self.p1, p2=self.p2, a1=self.a1)
        derivative = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        # Dispersion derivative is the analytic second temporal derivative,
        # matching R's HRF_SPMG3 which uses hrf_spmg1_second_deriv.
        dispersion = spmg1_second_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        result = np.column_stack([canonical, derivative, dispersion])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        deriv1 = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        deriv2 = spmg1_second_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        # Derivative of the dispersion basis = derivative of the analytic
        # second derivative; computed by central difference.
        dx = self._DISPERSION_DERIV_DX
        deriv3 = (
            spmg1_second_derivative(t + dx, p1=self.p1, p2=self.p2, a1=self.a1)
            - spmg1_second_derivative(t - dx, p1=self.p1, p2=self.p2, a1=self.a1)
        ) / (2 * dx)
        result = np.column_stack([deriv1, deriv2, deriv3])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result
