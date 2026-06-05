"""Pre-defined HRF objects.

Each predefined HRF kind is a typed ``@dataclass`` subclass of
:class:`HRF` with named parameter fields, then the module-level
constants (``GAMMA_HRF``, ``HRF_SPMG1``, ...) are default instances of
those classes. The ``FunctionHRF`` indirection survives only as the
adapter for raw callables (see ``as_hrf``).

Parameters live exclusively as the typed dataclass fields; the old
inherited ``params`` / ``param_names`` dict mirror was removed by bead
``bd-01KRGCZJ6JAA4BKRTNQ91P2PE5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate

from ..hrf_dispatch import SimpleHRF
from .core import HRF, HrfParamValue
from .functions import (
    boxcar_hrf,
    bspline_hrf,
    fir_basis,
    fourier_hrf,
    gamma_hrf,
    gaussian_hrf,
    hrf_basis_lwu,
    hrf_half_cosine,
    hrf_ident,
    hrf_inv_logit,
    hrf_lwu,
    hrf_mexhat,
    hrf_sine,
    hrf_time,
    weighted_hrf,
)
from .spm_hrf import (
    SPMG1_HRF,
    SPMG2_HRF,
    SPMG3_HRF,
    SPMG1_HRF_Legacy,
    SPMG2_HRF_Legacy,
    SPMG3_HRF_Legacy,
)

# --- Single-basis kernels --------------------------------------------------


@dataclass
class GammaHRF(HRF):
    name: str = "gamma"
    nbasis: int = 1
    span: float = 24.0
    shape: float = 6.0
    rate: float = 1.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return gamma_hrf(t, shape=self.shape, rate=self.rate)


@dataclass
class GaussianHRF(HRF):
    name: str = "gaussian"
    nbasis: int = 1
    span: float = 24.0
    mean: float = 6.0
    sd: float = 2.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return gaussian_hrf(t, mean=self.mean, sd=self.sd)


@dataclass
class MexhatHRF(HRF):
    name: str = "mexhat"
    nbasis: int = 1
    span: float = 24.0
    mean: float = 6.0
    sd: float = 2.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_mexhat(t, mean=self.mean, sd=self.sd)


@dataclass
class InvLogitHRF(HRF):
    name: str = "inv_logit"
    nbasis: int = 1
    span: float = 30.0
    mu1: float = 6.0
    s1: float = 1.0
    mu2: float = 16.0
    s2: float = 1.0
    lag: float = 0.0  # type: ignore[assignment]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_inv_logit(
            t, mu1=self.mu1, s1=self.s1, mu2=self.mu2, s2=self.s2, lag=self.lag,
        )


@dataclass
class HalfCosineHRF(HRF):
    name: str = "half_cosine"
    nbasis: int = 1
    span: float = 24.0
    h1: float = 1.0
    h2: float = 5.0
    h3: float = 7.0
    h4: float = 7.0
    f1: float = 0.0
    f2: float = 0.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_half_cosine(
            t, h1=self.h1, h2=self.h2, h3=self.h3, h4=self.h4,
            f1=self.f1, f2=self.f2,
        )


@dataclass
class TimeHRF(HRF):
    name: str = "time"
    nbasis: int = 1
    span: float = 22.0
    max_time: float = 22.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_time(t, max_time=self.max_time)


# --- Multi-basis kernels ---------------------------------------------------


@dataclass
class BSplineHRF(HRF):
    name: str = "bspline"
    nbasis: int = 5
    span: float = 24.0
    degree: int = 3

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return bspline_hrf(t, n_basis=self.nbasis, degree=self.degree, span=self.span)


@dataclass
class FIRHRF(HRF):
    name: str = "fir"
    nbasis: int = 12
    span: float = 24.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return fir_basis(t, n_basis=self.nbasis, span=self.span)


@dataclass
class FourierHRF(HRF):
    name: str = "fourier"
    nbasis: int = 5
    span: float = 24.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return fourier_hrf(t, n_basis=self.nbasis, span=self.span)


@dataclass
class SineHRF(HRF):
    name: str = "sine"
    nbasis: int = 5
    span: float = 24.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_sine(t, span=self.span, n_basis=self.nbasis)


@dataclass
class DaguerreHRF(HRF):
    """Daguerre basis HRF (orthogonal polynomials on [0, infinity))."""

    name: str = "daguerre"
    nbasis: int = 3
    span: float = 24.0
    scale: float = 4.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        from .functions import daguerre_basis
        return daguerre_basis(t, n_basis=self.nbasis, scale=self.scale)


@dataclass(init=False)
class LWUHRF(HRF):
    """Lag-Width-Undershoot HRF.

    Normalization is a separate typed wrapper; construct ``LWUHRF`` and
    pass it to :func:`fmrimod.hrf.normalize` when scaling is needed.
    """

    name: str = "lwu"
    nbasis: int = 1
    span: float = 30.0
    tau: float = 6.0
    sigma: float = 2.5
    rho: float = 0.35
    _legacy_normalize: Literal["none", "height", "area"] = field(
        default="none", init=False, repr=False
    )

    def __init__(
        self,
        name: str = "lwu",
        nbasis: int = 1,
        span: float = 30.0,
        tau: float = 6.0,
        sigma: float = 2.5,
        rho: float = 0.35,
        normalize: Literal["none", "height", "area"] = "none",
    ) -> None:
        self.name = name
        self.nbasis = nbasis
        self.span = span
        self.tau = tau
        self.sigma = sigma
        self.rho = rho
        self._legacy_normalize = normalize
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.sigma <= 0.05:
            raise ValueError("sigma must be > 0.05")
        if not 0 <= self.rho <= 1.5:
            raise ValueError("rho must be between 0 and 1.5")
        if self._legacy_normalize == "area":
            raise ValueError(
                "normalize='area' on LWUHRF is retired; "
                "use normalize(LWUHRF(...), 'unit_integral')"
            )
        if self._legacy_normalize == "height":
            raise ValueError(
                "normalize='height' on LWUHRF is retired; "
                "use normalize(LWUHRF(...), 'unit_peak')"
            )
        if self._legacy_normalize != "none":
            raise ValueError("normalize must be 'none', 'height', or 'area'")

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_lwu(
            t, tau=self.tau, sigma=self.sigma, rho=self.rho,
            normalize=self._legacy_normalize,
        )


def _default_lwu_theta0() -> tuple[float, float, float]:
    return (6.0, 2.5, 0.35)


@dataclass
class LWUBasisHRF(HRF):
    name: str = "lwu_basis"
    nbasis: int = 4
    span: float = 30.0
    theta0: tuple[float, float, float] = field(default_factory=_default_lwu_theta0)
    normalize_primary: Literal["none", "height"] = "none"

    def __post_init__(self) -> None:
        if len(self.theta0) != 3:
            raise ValueError("theta0 must have length 3 [tau, sigma, rho]")
        if self.normalize_primary == "height":
            raise ValueError(
                "normalize_primary='height' on LWUBasisHRF is retired; "
                "use normalize(LWUBasisHRF(...), 'unit_peak')"
            )
        if self.normalize_primary != "none":
            raise ValueError("normalize_primary must be 'none' or 'height'")

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_basis_lwu(
            list(self.theta0), t, normalize_primary=self.normalize_primary,
        )


# --- Adapter / data-driven kinds ------------------------------------------


@dataclass
class IdentityHRF(HRF):
    """Identity (sampled Dirac) HRF: 1 at ``t == 0``, 0 elsewhere."""

    name: str = "identity"
    nbasis: int = 1
    span: float = 1.0

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_ident(t)


@dataclass
class BoxcarHRF(HRF):
    """Boxcar (step function) HRF parameterised by ``width`` and ``amplitude``.

    The default ``name`` mirrors the legacy ``boxcar_generator`` format
    (``"boxcar[<width>]"``) so callers that pin the name continue to work.
    Set ``name`` explicitly to override.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 1.0
    width: float = 1.0
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("BoxcarHRF.width must be positive")
        # span tracks the boxcar duration unless the caller pinned it.
        if self.span == 1.0:
            self.span = float(self.width)
        if not self.name:
            self.name = f"boxcar[{self.width:.2g}]"

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return boxcar_hrf(t, width=self.width, amplitude=self.amplitude)


@dataclass
class WeightedHRF(HRF):
    """User-weighted HRF over an explicit time grid.

    The typed shape carries ``weights``, ``times``, and ``method`` directly;
    the ``weighted_generator`` adapter accepts the legacy
    ``width``/``weights`` form and resolves it to ``times`` before
    construction.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 0.0
    weights: tuple[float, ...] = (1.0, 1.0)
    times: tuple[float, ...] = (0.0, 1.0)
    method: Literal["constant", "linear"] = "constant"

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        times = np.asarray(self.times, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("WeightedHRF weights must be 1-D")
        if times.ndim != 1:
            raise ValueError("WeightedHRF times must be 1-D")
        if len(weights) < 2:
            raise ValueError("WeightedHRF requires at least 2 weights")
        if len(weights) != len(times):
            raise ValueError("WeightedHRF weights and times must align")
        if not np.all(np.diff(times) > 0):
            raise ValueError("WeightedHRF times must be strictly increasing")
        if times[0] < 0:
            raise ValueError("WeightedHRF times must start at 0 or later")
        if self.method not in ("constant", "linear"):
            raise ValueError("WeightedHRF.method must be 'constant' or 'linear'")
        self.weights = tuple(float(x) for x in weights)
        self.times = tuple(float(x) for x in times)
        if self.span == 0.0:
            self.span = float(times[-1])
        if not self.name:
            self.name = f"weighted[w={self.span:.2g}, {len(self.weights)} wts]"

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return weighted_hrf(
            t,
            weights=self.weights,
            times=self.times,
            method=self.method,
        )


@dataclass(init=False)
class EmpiricalHRF(HRF):
    """HRF interpolated from observed (t, y) sample points.

    The interpolator is built once at construction and reused on every
    call. ``t_points`` and ``y_values`` are stored as tuples so the
    instance is hashable-by-identity but inspectable by callers that need
    to introspect the underlying sample.
    """

    name: str = "empirical_hrf"
    nbasis: int = 1
    span: float = 1.0
    t_points: tuple[float, ...] = ()
    y_values: tuple[float, ...] = ()

    def __init__(
        self,
        t_points: ArrayLike,
        y_values: ArrayLike,
        name: str = "empirical_hrf",
        nbasis: int = 1,
        span: Optional[float] = None,
    ) -> None:
        t_arr = np.asarray(t_points, dtype=np.float64)
        y_arr = np.asarray(y_values, dtype=np.float64)
        if t_arr.shape != y_arr.shape:
            raise ValueError("EmpiricalHRF t_points and y_values must share shape")
        if t_arr.ndim != 1:
            raise ValueError("EmpiricalHRF t_points must be 1-D")
        if t_arr.size < 2:
            raise ValueError("EmpiricalHRF requires at least 2 sample points")

        order = np.argsort(t_arr)
        t_sorted = t_arr[order]
        y_sorted = y_arr[order]

        self.name = name
        self.nbasis = nbasis
        self.span = float(t_sorted[-1]) if span is None else float(span)
        self.t_points = tuple(t_sorted.tolist())
        self.y_values = tuple(y_sorted.tolist())
        self._interp = interpolate.interp1d(
            t_sorted,
            y_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        t_arr = np.asarray(t, dtype=np.float64)
        return np.asarray(self._interp(t_arr), dtype=np.float64)


# --- Singleton instances ---------------------------------------------------

# Default SPM HRF basis using the SPM/Nilearn standard parameterization
# (delay=6, undershoot=16, dispersion=1, u_dispersion=1, ratio=0.167).
# See ``fmrimod/hrf/spm_hrf.py`` for the migration note.
SPM_CANONICAL = SPMG1_HRF()
SPM_WITH_DERIVATIVE = SPMG2_HRF()
SPM_WITH_DISPERSION = SPMG3_HRF()

# Legacy R-fmrireg parameterization, reachable via ``basis="spm_legacy"``
# / ``"spmg1_legacy"`` / ``"spmg2_legacy"`` / ``"spmg3_legacy"``.
SPM_CANONICAL_LEGACY = SPMG1_HRF_Legacy()
SPM_WITH_DERIVATIVE_LEGACY = SPMG2_HRF_Legacy()
SPM_WITH_DISPERSION_LEGACY = SPMG3_HRF_Legacy()

GAMMA_HRF = GammaHRF()
GAUSSIAN_HRF = GaussianHRF()
BSPLINE_HRF = BSplineHRF()
FIR_HRF = FIRHRF()
FOURIER_HRF = FourierHRF()
TIME_HRF = TimeHRF()
MEXHAT_HRF = MexhatHRF()
INV_LOGIT_HRF = InvLogitHRF()
HALF_COSINE_HRF = HalfCosineHRF()
SINE_HRF = SineHRF()
LWU_HRF = LWUHRF()
LWU_BASIS_HRF = LWUBasisHRF()
IDENTITY_HRF = IdentityHRF()
BOXCAR_HRF = BoxcarHRF()
WEIGHTED_HRF = WeightedHRF()

# Simple HRF for testing
SIMPLE_HRF = SimpleHRF()

# Create a dictionary of all pre-defined HRFs
PREDEFINED_HRFS = {
    "simple": SIMPLE_HRF,
    "spmg1": SPM_CANONICAL,
    "spm": SPM_CANONICAL,
    "spm_canonical": SPM_CANONICAL,
    "spmg2": SPM_WITH_DERIVATIVE,
    "spmg3": SPM_WITH_DISPERSION,
    # Legacy R-fmrireg parameterization (pre-2026 alignment).
    "spm_legacy": SPM_CANONICAL_LEGACY,
    "spmg1_legacy": SPM_CANONICAL_LEGACY,
    "spm_canonical_legacy": SPM_CANONICAL_LEGACY,
    "spmg2_legacy": SPM_WITH_DERIVATIVE_LEGACY,
    "spmg3_legacy": SPM_WITH_DISPERSION_LEGACY,
    "gamma": GAMMA_HRF,
    "gaussian": GAUSSIAN_HRF,
    "bspline": BSPLINE_HRF,
    "fir": FIR_HRF,
    "fourier": FOURIER_HRF,
    "time": TIME_HRF,
    "mexhat": MEXHAT_HRF,
    "inv_logit": INV_LOGIT_HRF,
    "half_cosine": HALF_COSINE_HRF,
    "sine": SINE_HRF,
    "lwu": LWU_HRF,
    "lwu_basis": LWU_BASIS_HRF,
    "identity": IDENTITY_HRF,
    "ident": IDENTITY_HRF,
    "boxcar": BOXCAR_HRF,
    "weighted": WEIGHTED_HRF,
}


# R-parity aliases: match the fmrihrf constant names exactly so code ported
# from R reads identically. These are object aliases, not copies, so the same
# HRF instance is used regardless of which name is referenced.
HRF_SPMG1 = SPM_CANONICAL
HRF_SPMG2 = SPM_WITH_DERIVATIVE
HRF_SPMG3 = SPM_WITH_DISPERSION
HRF_GAMMA = GAMMA_HRF
HRF_GAUSSIAN = GAUSSIAN_HRF
HRF_BSPLINE = BSPLINE_HRF
HRF_FIR = FIR_HRF
HRF_FOURIER = FOURIER_HRF
HRF_TIME = TIME_HRF
HRF_MEXHAT = MEXHAT_HRF
HRF_INV_LOGIT = INV_LOGIT_HRF
HRF_HALF_COSINE = HALF_COSINE_HRF
HRF_SINE = SINE_HRF
HRF_LWU = LWU_HRF
HRF_LWU_BASIS = LWU_BASIS_HRF
HRF_IDENTITY = IDENTITY_HRF
HRF_IDENT = IDENTITY_HRF
HRF_BOXCAR = BOXCAR_HRF
HRF_WEIGHTED = WEIGHTED_HRF
