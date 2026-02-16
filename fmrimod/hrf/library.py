"""Pre-defined HRF objects."""

from __future__ import annotations

from .core import FunctionHRF
from .functions import (
    spm_canonical,
    gamma_hrf,
    gaussian_hrf,
    bspline_hrf,
    fourier_hrf,
    fir_basis,
    hrf_time,
    hrf_mexhat,
    hrf_inv_logit,
    hrf_half_cosine,
    hrf_sine,
    hrf_lwu,
    hrf_basis_lwu,
)
from .spm_hrf import SPMG1_HRF, SPMG2_HRF, SPMG3_HRF
from ..hrf_dispatch import SimpleHRF


# SPM Canonical HRF (now with analytic derivative support)
SPM_CANONICAL = SPMG1_HRF(p1=5.0, p2=15.0, a1=0.0833)

# SPM with temporal derivative (now with analytic derivative support)
SPM_WITH_DERIVATIVE = SPMG2_HRF(p1=5.0, p2=15.0, a1=0.0833)


# SPM with temporal and dispersion derivatives (now with analytic derivative support)
SPM_WITH_DISPERSION = SPMG3_HRF(p1=5.0, p2=15.0, a1=0.0833)


# Gamma HRF
GAMMA_HRF = FunctionHRF(
    func=lambda t: gamma_hrf(t, shape=6.0, rate=1.0),
    name="gamma",
    nbasis=1,
    span=24.0,
    params={"shape": 6.0, "rate": 1.0},
    param_names=["shape", "rate"],
)

# Gaussian HRF
GAUSSIAN_HRF = FunctionHRF(
    func=lambda t: gaussian_hrf(t, mean=6.0, sd=2.0),
    name="gaussian",
    nbasis=1,
    span=24.0,
    params={"mean": 6.0, "sd": 2.0},
    param_names=["mean", "sd"],
)

# B-spline basis HRF
BSPLINE_HRF = FunctionHRF(
    func=lambda t: bspline_hrf(t, n_basis=5, degree=3, span=24.0),
    name="bspline",
    nbasis=5,
    span=24.0,
    params={"n_basis": 5, "degree": 3},
    param_names=["n_basis", "degree"],
)

# FIR basis HRF
FIR_HRF = FunctionHRF(
    func=lambda t: fir_basis(t, n_basis=12, span=24.0),
    name="fir",
    nbasis=12,
    span=24.0,
    params={"n_basis": 12},
    param_names=["n_basis"],
)

# Fourier basis HRF
FOURIER_HRF = FunctionHRF(
    func=lambda t: fourier_hrf(t, n_basis=5, span=24.0),
    name="fourier",
    nbasis=5,
    span=24.0,
    params={"n_basis": 5},
    param_names=["n_basis"],
)

# Time HRF
TIME_HRF = FunctionHRF(
    func=lambda t: hrf_time(t, max_time=22.0),
    name="time",
    nbasis=1,
    span=22.0,
    params={"max_time": 22.0},
    param_names=["max_time"],
)

# Mexican Hat HRF
MEXHAT_HRF = FunctionHRF(
    func=lambda t: hrf_mexhat(t, mean=6.0, sd=2.0),
    name="mexhat",
    nbasis=1,
    span=24.0,
    params={"mean": 6.0, "sd": 2.0},
    param_names=["mean", "sd"],
)

# Inverse Logit HRF
INV_LOGIT_HRF = FunctionHRF(
    func=lambda t: hrf_inv_logit(t, mu1=6.0, s1=1.0, mu2=16.0, s2=1.0, lag=0.0),
    name="inv_logit",
    nbasis=1,
    span=30.0,
    params={"mu1": 6.0, "s1": 1.0, "mu2": 16.0, "s2": 1.0, "lag": 0.0},
    param_names=["mu1", "s1", "mu2", "s2", "lag"],
)

# Half Cosine HRF
HALF_COSINE_HRF = FunctionHRF(
    func=lambda t: hrf_half_cosine(t, h1=1.0, h2=5.0, h3=7.0, h4=7.0, f1=0.0, f2=0.0),
    name="half_cosine",
    nbasis=1,
    span=24.0,
    params={"h1": 1.0, "h2": 5.0, "h3": 7.0, "h4": 7.0, "f1": 0.0, "f2": 0.0},
    param_names=["h1", "h2", "h3", "h4", "f1", "f2"],
)

# Sine basis HRF
SINE_HRF = FunctionHRF(
    func=lambda t: hrf_sine(t, span=24.0, n_basis=5),
    name="sine",
    nbasis=5,
    span=24.0,
    params={"span": 24.0, "n_basis": 5},
    param_names=["span", "n_basis"],
)

# LWU HRF
LWU_HRF = FunctionHRF(
    func=lambda t: hrf_lwu(t, tau=6.0, sigma=2.5, rho=0.35, normalize="none"),
    name="lwu",
    nbasis=1,
    span=30.0,
    params={"tau": 6.0, "sigma": 2.5, "rho": 0.35, "normalize": "none"},
    param_names=["tau", "sigma", "rho", "normalize"],
)

# LWU Basis HRF
LWU_BASIS_HRF = FunctionHRF(
    func=lambda t: hrf_basis_lwu([6.0, 2.5, 0.35], t, normalize_primary="none"),
    name="lwu_basis",
    nbasis=4,
    span=30.0,
    params={"theta0": [6.0, 2.5, 0.35], "normalize_primary": "none"},
    param_names=["theta0", "normalize_primary"],
)

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
