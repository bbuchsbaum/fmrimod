"""SPM HRF implementations with finite-difference and legacy derivatives.

Each SPMG kind exposes its parameters as named typed fields. The
default classes use the SPM standard parameterization
(``delay=6, undershoot=16, dispersion=1, u_dispersion=1, ratio=0.167``)
matching SPM ``spm_hrf`` and Nilearn ``_gamma_difference_hrf``. The
legacy R-fmrireg parameterization (``p1=5, p2=15, a1=0.0833``) is
preserved on the ``*_HRF_Legacy`` siblings, reachable from the typed
spec via ``basis="spm_legacy"`` / ``"spmg1_legacy"`` / ``"spmg2_legacy"``
/ ``"spmg3_legacy"``.

Migration notes
---------------
- The default ``basis="spm"`` / ``"spmg1"`` / ``"spmg2"`` / ``"spmg3"``
  changed in early 2026 from the legacy R parameterization to the SPM
  standard form. The realised canonical column changed shape
  (peak/undershoot magnitude ratio went from ~4600 to the SPM-standard
  ~10), and the realised columns are now Pearson-correlated > 0.999
  with Nilearn's realised regressors on identical events.
- SPMG2's temporal derivative and SPMG3's dispersion derivative on
  the new classes use SPM's finite-difference scheme in the *delay*
  and *dispersion* parameters respectively, matching Nilearn's
  ``spm_time_derivative`` / ``spm_dispersion_derivative``. The legacy
  classes keep the analytic time-derivative and the dispersion-as-
  time-scaling derivative used previously.
- The old inherited ``params`` / ``param_names`` dict mirror was
  removed by bead ``bd-01KRGCZJ6JAA4BKRTNQ91P2PE5``. Evaluation reads
  the dataclass fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF
from .derivatives import (
    spmg1_derivative,
    spmg1_dispersion_derivative,
    spmg1_dispersion_derivative_spm,
    spmg1_second_derivative,
    spmg1_temporal_derivative_spm,
)
from .functions import (
    _SPM_DEFAULT_DELAY,
    _SPM_DEFAULT_DISPERSION,
    _SPM_DEFAULT_RATIO,
    _SPM_DEFAULT_U_DISPERSION,
    _SPM_DEFAULT_UNDERSHOOT,
    spm_canonical,
    spm_canonical_legacy,
)


# ---------------------------------------------------------------------------
# Default (SPM standard) classes
# ---------------------------------------------------------------------------

@dataclass
class SPMG1_HRF(HRF):
    """SPM canonical HRF (standard SPM/Nilearn parameterization).

    See :func:`spm_canonical` for the formula and default values.
    """

    name: str = "SPMG1"
    nbasis: int = 1
    span: float = 32.0
    delay: float = _SPM_DEFAULT_DELAY
    undershoot: float = _SPM_DEFAULT_UNDERSHOOT
    dispersion: float = _SPM_DEFAULT_DISPERSION
    u_dispersion: float = _SPM_DEFAULT_U_DISPERSION
    ratio: float = _SPM_DEFAULT_RATIO

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return spm_canonical(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )


@dataclass
class SPMG2_HRF(HRF):
    """SPM informed basis: canonical + SPM time derivative.

    The second column is the SPM/Nilearn time derivative, computed as
    a finite difference in the delay parameter:
    ``(h(delay=6) - h(delay=6+0.1)) / 0.1`` (matches SPM
    ``spm_get_bf``'s "hrf (with time derivative)" case and Nilearn's
    ``spm_time_derivative``).
    """

    name: str = "SPMG2"
    nbasis: int = 2
    span: float = 32.0
    delay: float = _SPM_DEFAULT_DELAY
    undershoot: float = _SPM_DEFAULT_UNDERSHOOT
    dispersion: float = _SPM_DEFAULT_DISPERSION
    u_dispersion: float = _SPM_DEFAULT_U_DISPERSION
    ratio: float = _SPM_DEFAULT_RATIO

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )
        derivative = spmg1_temporal_derivative_spm(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )
        result = np.column_stack([canonical, derivative])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result


@dataclass
class SPMG3_HRF(HRF):
    """SPM informed basis: canonical + temporal + dispersion derivatives.

    Both derivatives use SPM's finite-difference scheme in the
    relevant parameter (delay for column 2, dispersion for column 3),
    matching SPM's ``spm_get_bf`` "hrf (with time and dispersion
    derivatives)" and Nilearn's ``spm_time_derivative`` /
    ``spm_dispersion_derivative``.

    The R-side ``fmrireg::HRF_SPMG3`` historically used the second
    *time* derivative as the third column and a different canonical
    parameterization; fmrimod now diverges from that to match the
    SPM literature. The legacy form is preserved as
    :class:`SPMG3_HRF_Legacy` (basis ``"spmg3_legacy"``).
    """

    name: str = "SPMG3"
    nbasis: int = 3
    span: float = 32.0
    delay: float = _SPM_DEFAULT_DELAY
    undershoot: float = _SPM_DEFAULT_UNDERSHOOT
    dispersion: float = _SPM_DEFAULT_DISPERSION
    u_dispersion: float = _SPM_DEFAULT_U_DISPERSION
    ratio: float = _SPM_DEFAULT_RATIO

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )
        time_deriv = spmg1_temporal_derivative_spm(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )
        disp_deriv = spmg1_dispersion_derivative_spm(
            t, delay=self.delay, undershoot=self.undershoot,
            dispersion=self.dispersion, u_dispersion=self.u_dispersion,
            ratio=self.ratio,
        )
        result = np.column_stack([canonical, time_deriv, disp_deriv])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result


# ---------------------------------------------------------------------------
# Legacy classes (R fmrireg parameterization)
# ---------------------------------------------------------------------------

@dataclass
class SPMG1_HRF_Legacy(HRF):
    """Legacy SPM canonical HRF (R ``fmrireg`` parameterization)."""

    name: str = "SPMG1_legacy"
    nbasis: int = 1
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return spm_canonical_legacy(t, p1=self.p1, p2=self.p2, a1=self.a1)

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        return spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)


@dataclass
class SPMG2_HRF_Legacy(HRF):
    """Legacy SPMG2: legacy canonical + analytic ∂h/∂t."""

    name: str = "SPMG2_legacy"
    nbasis: int = 2
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical_legacy(t, p1=self.p1, p2=self.p2, a1=self.a1)
        derivative = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        result = np.column_stack([canonical, derivative])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        first_deriv = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        second_deriv = spmg1_second_derivative(
            t, p1=self.p1, p2=self.p2, a1=self.a1
        )
        result = np.column_stack([first_deriv, second_deriv])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result


@dataclass
class SPMG3_HRF_Legacy(HRF):
    """Legacy SPMG3 used by R ``fmrireg::HRF_SPMG3``.

    Third column is the *dispersion derivative computed on the legacy
    canonical* (finite difference w.r.t. a time-scaling ``dispersion``
    parameter on the legacy form), as introduced in the dispersion-
    derivative fix that preceded the SPM-canonical alignment. This is
    still numerically distinct from R fmrireg's
    ``hrf_spmg1_second_deriv`` — see the tracking issue on bbuchsbaum/
    fmrihrf for the R-side fix.
    """

    name: str = "SPMG3_legacy"
    nbasis: int = 3
    span: float = 24.0
    p1: float = 5.0
    p2: float = 15.0
    a1: float = 0.0833

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t)
        canonical = spm_canonical_legacy(t, p1=self.p1, p2=self.p2, a1=self.a1)
        derivative = spmg1_derivative(t, p1=self.p1, p2=self.p2, a1=self.a1)
        dispersion = spmg1_dispersion_derivative(
            t, p1=self.p1, p2=self.p2, a1=self.a1
        )
        result = np.column_stack([canonical, derivative, dispersion])
        if t.ndim == 0 or (t.ndim == 1 and len(t) == 1):
            result = result.reshape(1, -1)
        return result
