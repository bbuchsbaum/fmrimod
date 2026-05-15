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

from .core import HRF, HrfParamValue
from .derivatives import spmg1_derivative, spmg1_second_derivative
from .functions import spm_canonical

_SPM_PARAM_NAMES = ("p1", "p2", "a1")


def _spm_params(p1: float, p2: float, a1: float) -> dict[str, HrfParamValue]:
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
    """SPM HRF with canonical + temporal derivative basis (informed basis set).

    Two-column basis: column 0 is the SPM canonical HRF; column 1 is its
    **closed-form** first temporal derivative ``spmg1_derivative``.

    Divergence from Nilearn / SPM12
    -------------------------------
    SPM12 and Nilearn (``hrf_model="spm + derivative"``) compute the
    derivative column as a finite difference of the canonical HRF
    evaluated at the sampling grid: ``(h(t + dt) - h(t)) / dt`` with
    ``dt`` tied to the microtime resolution. We use the closed-form
    analytic derivative instead. The two agree on peak shape (Pearson
    ~0.98 in the ``tier_a_spm_derivative_basis`` parity workflow) but
    disagree on the rank-order of near-zero tail values because finite
    differences inject discretisation noise that the analytic form does
    not.

    The trade-off is intentional:

    - Closed-form is numerically cleaner and grid-independent; Nilearn /
      SPM12 finite-difference output depends on the chosen ``dt``.
    - Friston et al.'s Taylor-expansion justification for the informed
      basis set (``h(t - Δt) ≈ h(t) - Δt · h'(t)``) uses the analytic
      derivative; we implement what the theory says, while SPM12
      implements a finite-difference approximation to it.

    **Downstream calibration caveat.** Latency-shift estimates derived
    from ``β_derivative / β_canonical`` (Henson 2002, Liao 2002) carry a
    proportionality constant that depends on the derivative
    implementation. The constants published in the SPM12-anchored
    literature are calibrated to the finite-difference column; they do
    not transfer to ``β`` ratios from this HRF without re-derivation.
    Bit-compatibility with SPM12-published latency calibration is an
    explicit non-goal here.

    See ``benchmarks/parity/tier_a_spm_derivative_basis/workflow.py`` for
    the parity case that exercises this divergence and the tolerance
    posture it forces (Pearson + max_abs as the gate; Spearman dropped
    because near-zero tail rank-order is implementation-dependent).
    """

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
    """SPM HRF with canonical + temporal + dispersion derivatives.

    Three-column basis: column 0 is the SPM canonical HRF; column 1 is
    its **closed-form** first temporal derivative; column 2 is the
    closed-form second temporal derivative (used as the dispersion
    derivative, matching R's ``HRF_SPMG3`` via
    ``hrf_spmg1_second_deriv``).

    The same closed-form-vs-finite-difference divergence from Nilearn
    and SPM12 documented on :class:`SPMG2_HRF` applies to columns 1 and
    2 here. SPM12 and Nilearn (``hrf_model="spm + derivative +
    dispersion"``) finite-difference the canonical for the temporal
    derivative and finite-difference the canonical with a perturbed
    dispersion parameter for the dispersion derivative; we evaluate the
    closed-form first and second temporal derivatives directly.

    Latency-shift / dispersion-shift estimates derived from
    ``β_d / β_canonical`` ratios are not bit-compatible with
    SPM12-published calibrations — see the SPMG2_HRF docstring.
    """

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
        self.params = _spm_params(self.p1, self.p2, self.a1)
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
