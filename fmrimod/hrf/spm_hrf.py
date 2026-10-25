"""SPM canonical HRF and its informed basis sets, as R fmrihrf defines them.

The canonical is the double gamma ``gamma(delay) - ratio * gamma(undershoot)``
(:func:`~fmrimod.hrf.functions.spm_canonical`). At the defaults
(``delay=6, undershoot=16, dispersion=1, u_dispersion=1, ratio=1/6``,
``span=24``) it is R fmrihrf's ``HRF_SPMG1``,
``exp(-t) * (t^5 / 120 - t^15 / (6 * 15!))``, and SPM's ``spm_hrf``.

Informed bases follow R ``HRF_SPMG2`` / ``HRF_SPMG3``:

- column 2 is the analytic time derivative ``dh/dt``;
- column 3 (SPMG3) is SPM's dispersion difference, which perturbs only the
  peak gamma (identical in SPM, Nilearn and R).

Steward decision D2 (2026-10-25, decision note 0005) chose R's conventions.
Before it, fmrimod used Nilearn's ``ratio=0.167``, a 32 s span and SPM's
finite difference in the delay parameter for column 2, and kept the
pre-2026 R parameterization on ``*_HRF_Legacy`` classes; those are gone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from .derivatives import (
    spmg1_derivative,
    spmg1_dispersion_derivative,
    spmg1_dispersion_time_derivative,
)
from .functions import (
    _SPM_DEFAULT_DELAY,
    _SPM_DEFAULT_DISPERSION,
    _SPM_DEFAULT_RATIO,
    _SPM_DEFAULT_SPAN,
    _SPM_DEFAULT_U_DISPERSION,
    _SPM_DEFAULT_UNDERSHOOT,
    spm_canonical,
)


@dataclass
class _SPMFamily(HRF):
    """Shared SPM canonical parameters for SPMG1/2/3."""

    span: float = _SPM_DEFAULT_SPAN
    delay: float = _SPM_DEFAULT_DELAY
    undershoot: float = _SPM_DEFAULT_UNDERSHOOT
    dispersion: float = _SPM_DEFAULT_DISPERSION
    u_dispersion: float = _SPM_DEFAULT_U_DISPERSION
    ratio: float = _SPM_DEFAULT_RATIO

    def _params(self) -> dict[str, float]:
        return {
            "delay": self.delay,
            "undershoot": self.undershoot,
            "dispersion": self.dispersion,
            "u_dispersion": self.u_dispersion,
            "ratio": self.ratio,
        }


def _stack(
    t: NDArray[np.float64], *columns: NDArray[np.float64]
) -> NDArray[np.float64]:
    result = np.column_stack(columns)
    if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
        result = result.reshape(1, -1)
    return result


@dataclass
class SPMG1_HRF(_SPMFamily):
    """SPM canonical HRF (R ``HRF_SPMG1``); see :func:`spm_canonical`."""

    name: str = "SPMG1"
    nbasis: int = 1

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return spm_canonical(t, **self._params())

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """Analytic ``dh/dt`` (R ``deriv.SPMG1_HRF``)."""
        return spmg1_derivative(t, **self._params())


@dataclass
class SPMG2_HRF(_SPMFamily):
    """Canonical plus its analytic time derivative (R ``HRF_SPMG2``)."""

    name: str = "SPMG2"
    nbasis: int = 2

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        p = self._params()
        return _stack(t, spm_canonical(t, **p), spmg1_derivative(t, **p))

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """First and second time derivatives (R ``deriv.SPMG2_HRF``)."""
        t = np.asarray(t)
        p = self._params()
        return _stack(t, spmg1_derivative(t, **p), spmg1_derivative(t, order=2, **p))


@dataclass
class SPMG3_HRF(_SPMFamily):
    """Canonical, time derivative and dispersion column (R ``HRF_SPMG3``)."""

    name: str = "SPMG3"
    nbasis: int = 3

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        p = self._params()
        return _stack(
            t,
            spm_canonical(t, **p),
            spmg1_derivative(t, **p),
            spmg1_dispersion_derivative(t, **p),
        )

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        """R ``deriv.SPMG3_HRF``: d/dt and d2/dt2 of the canonical, then
        the time derivative of the dispersion column."""
        t = np.asarray(t)
        p = self._params()
        return _stack(
            t,
            spmg1_derivative(t, **p),
            spmg1_derivative(t, order=2, **p),
            spmg1_dispersion_time_derivative(t, **p),
        )
