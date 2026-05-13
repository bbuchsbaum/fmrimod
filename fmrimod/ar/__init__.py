"""Autoregressive noise modelling for fMRI time series."""

from .plan import WhiteningPlan, WhitenResult
from .numhelpers import (
    pacf_to_ar,
    ar_to_pacf,
    enforce_stationary_ar,
    enforce_invertible_ma,
    levinson_durbin,
    segmented_acvf,
    run_avg_acvf,
)
from .estimation import (
    estimate_ar,
    estimate_ar_yule_walker,
    estimate_ar_voxelwise,
    estimate_ar_bic,
    fit_noise,
)
from .hr_arma import hr_arma
from .whitening import (
    ar_whiten,
    ar_whiten_matrix,
    ar_covariance_matrix,
    arma_whiten_segments,
    whiten_apply,
    whiten,
)
from .multiscale import (
    parcel_means,
    ms_dispersion,
    ms_weights,
    ms_combine_to_fine,
    ms_parent_maps,
    ms_estimate_scale,
)
from .afni import afni_phi_ar3, afni_phi_ar5, afni_restricted_plan
from .diagnostics import acorr_diagnostics, sandwich_from_whitened_resid
from .compat import plan_from_phi, whiten_with_phi
from .integration import iterative_gls, iterative_ar_gls
from .nilearn_ar1 import (
    Ar1NilearnConfig,
    DEFAULT_BIN_WIDTH,
    ar1_nilearn,
    bin_ar1_coefficients,
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
