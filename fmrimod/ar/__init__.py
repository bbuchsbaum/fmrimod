"""Autoregressive noise modelling for fMRI time series."""

from .afni import afni_phi_ar3, afni_phi_ar5, afni_restricted_plan
from .diagnostics import acorr_diagnostics, sandwich_from_whitened_resid
from .estimation import (
    estimate_ar,
    estimate_ar_bic,
    estimate_ar_voxelwise,
    estimate_ar_yule_walker,
    fit_noise,
)
from .hr_arma import hr_arma
from .integration import iterative_ar_gls, iterative_gls
from .multiscale import (
    ms_combine_to_fine,
    ms_dispersion,
    ms_estimate_scale,
    ms_parent_maps,
    ms_weights,
    parcel_means,
)
from .nilearn_ar1 import (
    DEFAULT_BIN_WIDTH,
    Ar1NilearnConfig,
    ar1_nilearn,
    bin_ar1_coefficients,
)
from .numhelpers import (
    ar_to_pacf,
    enforce_invertible_ma,
    enforce_stationary_ar,
    levinson_durbin,
    pacf_to_ar,
    run_avg_acvf,
    segmented_acvf,
)
from .plan import WhiteningPlan, WhitenResult, plan_from_phi, whiten_with_phi
from .whitening import (
    ar_covariance_matrix,
    ar_whiten,
    ar_whiten_matrix,
    arma_whiten_segments,
    whiten,
    whiten_apply,
)

__all__ = [
    # Plan / result
    "WhiteningPlan",
    "WhitenResult",
    # Numeric helpers
    "pacf_to_ar",
    "ar_to_pacf",
    "enforce_stationary_ar",
    "enforce_invertible_ma",
    "levinson_durbin",
    "segmented_acvf",
    "run_avg_acvf",
    # Estimation
    "estimate_ar",
    "estimate_ar_yule_walker",
    "estimate_ar_voxelwise",
    "estimate_ar_bic",
    "fit_noise",
    # ARMA
    "hr_arma",
    # Whitening
    "ar_whiten",
    "ar_whiten_matrix",
    "ar_covariance_matrix",
    "arma_whiten_segments",
    "whiten_apply",
    "whiten",
    # Multi-scale
    "parcel_means",
    "ms_dispersion",
    "ms_weights",
    "ms_combine_to_fine",
    "ms_parent_maps",
    "ms_estimate_scale",
    # AFNI
    "afni_phi_ar3",
    "afni_phi_ar5",
    "afni_restricted_plan",
    # Diagnostics
    "acorr_diagnostics",
    "sandwich_from_whitened_resid",
    # Compat
    "plan_from_phi",
    "whiten_with_phi",
    # Integration
    "iterative_gls",
    "iterative_ar_gls",
    # Nilearn-compatible AR(1)
    "ar1_nilearn",
    "Ar1NilearnConfig",
    "bin_ar1_coefficients",
    "DEFAULT_BIN_WIDTH",
]
