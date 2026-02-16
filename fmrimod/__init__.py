"""fmrimod - fMRI Signal Modeling: HRFs, Design Matrices, and Regression.

A unified Python library for fMRI experimental design and signal modeling,
combining hemodynamic response function (HRF) specification, design matrix
construction, regression, and GLM fitting tools.

Subpackages
-----------
hrf : HRF basis functions, registry, decorators, and generators
regressor : Event-related regressors and convolution
events : Event representations (factor, variable, matrix, basis)
formula : R-style formula parsing and DSL
basis : Parametric basis functions (polynomial, spline, transforms)
contrast : Contrast specification and F-tests
baseline : Baseline and nuisance models
design : Design matrix assembly and EventModel
visualization : Design matrix plotting and correlation maps
utils : Generic functions, term/event utilities, caching
dataset : Dataset abstractions and adapters for fMRI data
model : FmriModel combining event model, baseline, and dataset
glm : GLM fitting engine (OLS, GLS, contrasts)
ar : Autoregressive noise modelling and whitening
robust : Robust regression via IRLS (Huber, bisquare)
stats : Statistical inference (p-values, FDR, spatial FDR)
simulate : BOLD simulation and noise generation
backends : Solver backends (numpy default, optional JAX)
lowrank : Sketch-based low-rank GLM solver (Gaussian, SRHT, CountSketch, Nyström)
betas : Trial-wise beta extraction (OLS, LSS)
bids : BIDS-Stats-Model export
"""

__version__ = "0.1.0"

# ── Core classes ──────────────────────────────────────────────────────
from .hrf.core import HRF
from .hrf_dispatch import as_hrf
from .sampling import SamplingFrame
from .regressor import regressor, regressor_set, null_regressor

# ── Pre-defined HRFs (most commonly used) ────────────────────────────
from .hrf.library import SPM_CANONICAL, SPM_WITH_DERIVATIVE, SPM_WITH_DISPERSION
from .hrf.registry import get_hrf, list_available_hrfs

# ── HRF generators ───────────────────────────────────────────────────
from .hrf.generators import gen_hrf, gen_hrf_set
from .hrf.decorators import lag_hrf, block_hrf

# ── Design (lazy imports to avoid circular dependencies) ─────────────
def event_model(*args, **kwargs):
    """Create an EventModel. See :func:`fmrimod.design.event_model.event_model`."""
    from .design.event_model import event_model as _event_model
    return _event_model(*args, **kwargs)

def design_matrix(*args, **kwargs):
    """Assemble a design matrix. See :func:`fmrimod.design.design_matrix.design_matrix`."""
    from .design.design_matrix import design_matrix as _design_matrix
    return _design_matrix(*args, **kwargs)

# ── Baseline ─────────────────────────────────────────────────────────
def baseline_model(*args, **kwargs):
    """Create a baseline model. See :func:`fmrimod.baseline.baseline_model`."""
    from .baseline import baseline_model as _baseline_model
    return _baseline_model(*args, **kwargs)

# ── Contrast ─────────────────────────────────────────────────────────
def contrast(*args, **kwargs):
    """Create a contrast. See :func:`fmrimod.contrast.contrast_spec.contrast`."""
    from .contrast.contrast_spec import contrast as _contrast
    return _contrast(*args, **kwargs)

# ── Basis functions ──────────────────────────────────────────────────
from .basis.polynomial import Poly
from .basis.transform import Scale, RobustScale
from .condition_basis import condition_basis_list

# ── Formula functions (lazy to avoid shadowing hrf subpackage) ───────
from functools import partial as _partial

def hrf_formula(*args, **kwargs):
    """HRF formula function. See :func:`fmrimod.formula.functional.hrf`."""
    from .formula.functional import hrf as _hrf_func
    return _hrf_func(*args, **kwargs)

hrf_spmg1 = _partial(hrf_formula, spec='spmg1')

# ── GLM fitting (lazy imports) ──────────────────────────────────────
def fmri_lm(*args, **kwargs):
    """Fit a GLM to fMRI data. See :func:`fmrimod.glm.fmri_lm.fmri_lm`."""
    from .glm.fmri_lm import fmri_lm as _fmri_lm
    return _fmri_lm(*args, **kwargs)

# ── Beta extraction (lazy) ────────────────────────────────────────
def estimate_betas(*args, **kwargs):
    """Estimate trial-wise betas. See :func:`fmrimod.betas.extraction.estimate_betas`."""
    from .betas.extraction import estimate_betas as _estimate_betas
    return _estimate_betas(*args, **kwargs)

# ── Bootstrap (lazy) ──────────────────────────────────────────────
def bootstrap_glm(*args, **kwargs):
    """Bootstrap CIs for GLM. See :func:`fmrimod.glm.bootstrap.bootstrap_glm`."""
    from .glm.bootstrap import bootstrap_glm as _bootstrap_glm
    return _bootstrap_glm(*args, **kwargs)

__all__ = [
    "__version__",
    # Core classes
    "HRF",
    "as_hrf",
    "SamplingFrame",
    # Regressors
    "regressor",
    "regressor_set",
    "null_regressor",
    # Pre-defined HRFs
    "SPM_CANONICAL",
    "SPM_WITH_DERIVATIVE",
    "SPM_WITH_DISPERSION",
    # HRF registry
    "get_hrf",
    "list_available_hrfs",
    # HRF generators & decorators
    "gen_hrf",
    "gen_hrf_set",
    "lag_hrf",
    "block_hrf",
    # Design (lazy)
    "event_model",
    "design_matrix",
    "baseline_model",
    "contrast",
    # Basis functions
    "Poly",
    "Scale",
    "RobustScale",
    "condition_basis_list",
    # Formula functions
    "hrf_formula",
    "hrf_spmg1",
    # GLM fitting
    "fmri_lm",
    # Beta extraction
    "estimate_betas",
    # Bootstrap
    "bootstrap_glm",
]
