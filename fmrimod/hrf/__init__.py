"""HRF module."""

from .core import HRF, FunctionHRF, as_hrf, bind_basis, hrf_from_coefficients
from .functions import (
    spm_canonical, gamma_hrf, gaussian_hrf,
    bspline_hrf, fourier_hrf, fir_basis,
    mexhat_hrf, sine_hrf, half_cosine_hrf, inv_logit_hrf, lwu_hrf,
    hrf_time, hrf_sine, hrf_mexhat, hrf_inv_logit, hrf_half_cosine,
    hrf_lwu, hrf_basis_lwu, hrf_ident,
)
from .library import (
    SPM_CANONICAL, SPM_WITH_DERIVATIVE, SPM_WITH_DISPERSION,
    GAMMA_HRF, GAUSSIAN_HRF, BSPLINE_HRF, FIR_HRF, FOURIER_HRF,
    TIME_HRF, MEXHAT_HRF, INV_LOGIT_HRF, HALF_COSINE_HRF, SINE_HRF,
    LWU_HRF, LWU_BASIS_HRF,
    PREDEFINED_HRFS,
    # R-parity aliases for fmrihrf constant names
    HRF_SPMG1, HRF_SPMG2, HRF_SPMG3,
    HRF_GAMMA, HRF_GAUSSIAN, HRF_BSPLINE, HRF_FIR, HRF_FOURIER,
    HRF_TIME, HRF_MEXHAT, HRF_INV_LOGIT, HRF_HALF_COSINE, HRF_SINE,
    HRF_LWU, HRF_LWU_BASIS,
)
from .decorators import (
    lag_hrf, block_hrf, normalize_hrf,
    gen_hrf_lagged, gen_hrf_blocked,
    hrf_lagged, hrf_blocked,
)
from .generators import (
    gen_hrf, gen_hrf_set, hrf_set, make_hrf,
    bspline_generator, fir_generator, fourier_generator, daguerre_generator,
    hrf_bspline_generator, hrf_fir_generator, hrf_fourier_generator,
    hrf_daguerre_generator, hrf_tent_generator,
)
from .registry import (
    get_hrf, list_available_hrfs, register_hrf, remove_hrf,
    clear_registry, _HRF_REGISTRY,
)
from .penalty import penalty_matrix
from .empirical import empirical_hrf, gen_empirical_hrf
from .hrf_library import hrf_library, gen_hrf_library
from .reconstruction import reconstruction_matrix
from .derivatives import deriv

__all__ = [
    # Core
    "HRF", "FunctionHRF", "as_hrf", "bind_basis", "hrf_from_coefficients",
    # Functions
    "spm_canonical", "gamma_hrf", "gaussian_hrf",
    "bspline_hrf", "fourier_hrf", "fir_basis",
    "mexhat_hrf", "sine_hrf", "half_cosine_hrf", "inv_logit_hrf", "lwu_hrf",
    "hrf_time", "hrf_sine", "hrf_mexhat", "hrf_inv_logit", "hrf_half_cosine",
    "hrf_lwu", "hrf_basis_lwu", "hrf_ident",
    # Pre-defined HRFs
    "SPM_CANONICAL", "SPM_WITH_DERIVATIVE", "SPM_WITH_DISPERSION",
    "GAMMA_HRF", "GAUSSIAN_HRF", "BSPLINE_HRF", "FIR_HRF", "FOURIER_HRF",
    "TIME_HRF", "MEXHAT_HRF", "INV_LOGIT_HRF", "HALF_COSINE_HRF", "SINE_HRF",
    "LWU_HRF", "LWU_BASIS_HRF",
    # R-parity aliases
    "HRF_SPMG1", "HRF_SPMG2", "HRF_SPMG3",
    "HRF_GAMMA", "HRF_GAUSSIAN", "HRF_BSPLINE", "HRF_FIR", "HRF_FOURIER",
    "HRF_TIME", "HRF_MEXHAT", "HRF_INV_LOGIT", "HRF_HALF_COSINE", "HRF_SINE",
    "HRF_LWU", "HRF_LWU_BASIS",
    # Decorators
    "lag_hrf", "block_hrf", "normalize_hrf",
    "gen_hrf_lagged", "gen_hrf_blocked",
    "hrf_lagged", "hrf_blocked",
    # Generators
    "gen_hrf", "gen_hrf_set", "hrf_set", "make_hrf",
    "bspline_generator", "fir_generator", "fourier_generator", "daguerre_generator",
    "hrf_bspline_generator", "hrf_fir_generator", "hrf_fourier_generator",
    "hrf_daguerre_generator", "hrf_tent_generator",
    # Registry
    "get_hrf", "list_available_hrfs", "register_hrf", "remove_hrf", "clear_registry",
    "_HRF_REGISTRY", "PREDEFINED_HRFS",
    # Penalty
    "penalty_matrix",
    # Empirical
    "empirical_hrf", "gen_empirical_hrf",
    # Library
    "hrf_library", "gen_hrf_library",
    # Reconstruction
    "reconstruction_matrix",
    # Derivatives
    "deriv",
]
