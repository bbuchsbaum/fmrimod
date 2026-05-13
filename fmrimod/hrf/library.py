"""Pre-defined HRF objects.

Each predefined HRF kind is a typed ``@dataclass`` subclass of
:class:`HRF` with named parameter fields, then the module-level
constants (``GAMMA_HRF``, ``HRF_SPMG1``, ...) are default instances of
those classes. The ``FunctionHRF`` indirection survives only as the
adapter for raw callables (see ``empirical_hrf``, ``as_hrf``).

During the transition window each class still mirrors its typed fields
into the inherited ``params`` / ``param_names`` for cross-testing
readers; bead ``bd-01KRGCZJ6JAA4BKRTNQ91P2PE5`` retires that mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..hrf_dispatch import SimpleHRF
from .core import HRF, HrfParamValue
from .functions import (
    bspline_hrf,
    fir_basis,
    fourier_hrf,
    gamma_hrf,
    gaussian_hrf,
    hrf_basis_lwu,
    hrf_half_cosine,
    hrf_inv_logit,
    hrf_lwu,
    hrf_mexhat,
    hrf_sine,
    hrf_time,
)
from .spm_hrf import SPMG1_HRF, SPMG2_HRF, SPMG3_HRF

# --- Single-basis kernels --------------------------------------------------


@dataclass
class GammaHRF(HRF):
    name: str = "gamma"
    nbasis: int = 1
    span: float = 24.0
    shape: float = 6.0
    rate: float = 1.0

    def __post_init__(self) -> None:
        self.params = {"shape": self.shape, "rate": self.rate}
        self.param_names = ["shape", "rate"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return gamma_hrf(t, shape=self.shape, rate=self.rate)


@dataclass
class GaussianHRF(HRF):
    name: str = "gaussian"
    nbasis: int = 1
    span: float = 24.0
    mean: float = 6.0
    sd: float = 2.0

    def __post_init__(self) -> None:
        self.params = {"mean": self.mean, "sd": self.sd}
        self.param_names = ["mean", "sd"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return gaussian_hrf(t, mean=self.mean, sd=self.sd)


@dataclass
class MexhatHRF(HRF):
    name: str = "mexhat"
    nbasis: int = 1
    span: float = 24.0
    mean: float = 6.0
    sd: float = 2.0

    def __post_init__(self) -> None:
        self.params = {"mean": self.mean, "sd": self.sd}
        self.param_names = ["mean", "sd"]

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
    lag: float = 0.0

    def __post_init__(self) -> None:
        self.params = {
            "mu1": self.mu1,
            "s1": self.s1,
            "mu2": self.mu2,
            "s2": self.s2,
            "lag": self.lag,
        }
        self.param_names = ["mu1", "s1", "mu2", "s2", "lag"]

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

    def __post_init__(self) -> None:
        self.params = {
            "h1": self.h1, "h2": self.h2, "h3": self.h3, "h4": self.h4,
            "f1": self.f1, "f2": self.f2,
        }
        self.param_names = ["h1", "h2", "h3", "h4", "f1", "f2"]

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

    def __post_init__(self) -> None:
        self.params = {"max_time": self.max_time}
        self.param_names = ["max_time"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_time(t, max_time=self.max_time)


# --- Multi-basis kernels ---------------------------------------------------


@dataclass
class BSplineHRF(HRF):
    name: str = "bspline"
    nbasis: int = 5
    span: float = 24.0
    degree: int = 3

    def __post_init__(self) -> None:
        self.params = {"n_basis": self.nbasis, "degree": self.degree}
        self.param_names = ["n_basis", "degree"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return bspline_hrf(t, n_basis=self.nbasis, degree=self.degree, span=self.span)


@dataclass
class FIRHRF(HRF):
    name: str = "fir"
    nbasis: int = 12
    span: float = 24.0

    def __post_init__(self) -> None:
        self.params = {"n_basis": self.nbasis}
        self.param_names = ["n_basis"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return fir_basis(t, n_basis=self.nbasis, span=self.span)


@dataclass
class FourierHRF(HRF):
    name: str = "fourier"
    nbasis: int = 5
    span: float = 24.0

    def __post_init__(self) -> None:
        self.params = {"n_basis": self.nbasis}
        self.param_names = ["n_basis"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return fourier_hrf(t, n_basis=self.nbasis, span=self.span)


@dataclass
class SineHRF(HRF):
    name: str = "sine"
    nbasis: int = 5
    span: float = 24.0

    def __post_init__(self) -> None:
        self.params = {"span": self.span, "n_basis": self.nbasis}
        self.param_names = ["span", "n_basis"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_sine(t, span=self.span, n_basis=self.nbasis)


@dataclass
class DaguerreHRF(HRF):
    """Daguerre basis HRF (orthogonal polynomials on [0, infinity))."""

    name: str = "daguerre"
    nbasis: int = 3
    span: float = 24.0
    scale: float = 4.0

    def __post_init__(self) -> None:
        self.params = {"n_basis": self.nbasis, "scale": self.scale}
        self.param_names = ["n_basis", "scale"]

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
        params: dict[str, HrfParamValue] | None = None,
        param_names: list[str] | None = None,
        tau: float = 6.0,
        sigma: float = 2.5,
        rho: float = 0.35,
        normalize: Literal["none", "height", "area"] = "none",
    ) -> None:
        self.name = name
        self.nbasis = nbasis
        self.span = span
        self.params = {} if params is None else dict(params)
        self.param_names = param_names
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
        self.params = {
            "tau": self.tau, "sigma": self.sigma, "rho": self.rho,
            "normalize": self._legacy_normalize,
        }
        self.param_names = ["tau", "sigma", "rho", "normalize"]

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
        # params back-compat keeps the list form ([..]) since cross_testing
        # readers may rely on it; the typed field is a tuple.
        self.params = {
            "theta0": list(self.theta0),
            "normalize_primary": self.normalize_primary,
        }
        self.param_names = ["theta0", "normalize_primary"]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        return hrf_basis_lwu(
            list(self.theta0), t, normalize_primary=self.normalize_primary,
        )


# --- Singleton instances ---------------------------------------------------

# SPM Canonical HRF (now with analytic derivative support)
SPM_CANONICAL = SPMG1_HRF(p1=5.0, p2=15.0, a1=0.0833)
SPM_WITH_DERIVATIVE = SPMG2_HRF(p1=5.0, p2=15.0, a1=0.0833)
SPM_WITH_DISPERSION = SPMG3_HRF(p1=5.0, p2=15.0, a1=0.0833)

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
