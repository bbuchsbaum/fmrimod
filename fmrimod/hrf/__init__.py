"""HRF module."""

from .core import HRF, FunctionHRF, as_hrf, bind_basis, hrf_from_coefficients
from .decorators import (
    block_hrf,
    gen_hrf_blocked,
    gen_hrf_lagged,
    hrf_blocked,
    hrf_lagged,
    lag_hrf,
)
from .derivatives import deriv
from .empirical import empirical_hrf, gen_empirical_hrf
from .functions import (
    bspline_hrf,
    fir_basis,
    fourier_hrf,
    gamma_hrf,
    gaussian_hrf,
    half_cosine_hrf,
    hrf_basis_lwu,
    hrf_half_cosine,
    hrf_ident,
    hrf_inv_logit,
    hrf_lwu,
    hrf_mexhat,
    hrf_sine,
    hrf_time,
    inv_logit_hrf,
    lwu_hrf,
    mexhat_hrf,
    sine_hrf,
    spm_canonical,
)
from .generators import (
    bspline_generator,
    daguerre_generator,
    fir_generator,
    fourier_generator,
    gen_hrf,
    gen_hrf_set,
    hrf_bspline_generator,
    hrf_daguerre_generator,
    hrf_fir_generator,
    hrf_fourier_generator,
    hrf_set,
    hrf_tent_generator,
    make_hrf,
)
from .hrf_library import gen_hrf_library, hrf_library
from .library import (
    BOXCAR_HRF,
    BSPLINE_HRF,
    FIR_HRF,
    FOURIER_HRF,
    GAMMA_HRF,
    GAUSSIAN_HRF,
    HALF_COSINE_HRF,
    HRF_BOXCAR,
    HRF_BSPLINE,
    HRF_FIR,
    HRF_FOURIER,
    HRF_GAMMA,
    HRF_GAUSSIAN,
    HRF_HALF_COSINE,
    HRF_IDENT,
    HRF_IDENTITY,
    HRF_INV_LOGIT,
    HRF_LWU,
    HRF_LWU_BASIS,
    HRF_MEXHAT,
    HRF_SINE,
    # R-parity aliases for fmrihrf constant names
    HRF_SPMG1,
    HRF_SPMG2,
    HRF_SPMG3,
    HRF_TIME,
    HRF_WEIGHTED,
    IDENTITY_HRF,
    INV_LOGIT_HRF,
    LWU_BASIS_HRF,
    LWU_HRF,
    MEXHAT_HRF,
    PREDEFINED_HRFS,
    SINE_HRF,
    SPM_CANONICAL,
    SPM_WITH_DERIVATIVE,
    SPM_WITH_DISPERSION,
    TIME_HRF,
    WEIGHTED_HRF,
    BoxcarHRF,
    EmpiricalHRF,
    IdentityHRF,
    WeightedHRF,
)
from .normalization import NormMode, normalize
from .penalty import penalty_matrix
from .reconstruction import reconstruction_matrix
from .registry import (
    _HRF_REGISTRY,
    clear_registry,
    get_hrf,
    list_available_hrfs,
    register_hrf,
    remove_hrf,
)

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
    "LWU_HRF", "LWU_BASIS_HRF", "IDENTITY_HRF", "BOXCAR_HRF", "WEIGHTED_HRF",
    "IdentityHRF", "BoxcarHRF", "WeightedHRF", "EmpiricalHRF",
    # R-parity aliases
    "HRF_SPMG1", "HRF_SPMG2", "HRF_SPMG3",
    "HRF_GAMMA", "HRF_GAUSSIAN", "HRF_BSPLINE", "HRF_FIR", "HRF_FOURIER",
    "HRF_TIME", "HRF_MEXHAT", "HRF_INV_LOGIT", "HRF_HALF_COSINE", "HRF_SINE",
    "HRF_LWU", "HRF_LWU_BASIS", "HRF_IDENTITY", "HRF_IDENT",
    "HRF_BOXCAR", "HRF_WEIGHTED",
    # Decorators
    "lag_hrf", "block_hrf",
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
    # Normalization modes
    "NormMode", "normalize",
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
