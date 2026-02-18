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


def event_factor(*args, **kwargs):
    """Construct a categorical event factor."""
    from .events import EventFactor
    return EventFactor(*args, **kwargs)


def event_variable(*args, **kwargs):
    """Construct a continuous event variable."""
    from .events import EventVariable
    return EventVariable(*args, **kwargs)


def event_matrix(*args, **kwargs):
    """Construct a multi-column event matrix."""
    from .events import EventMatrix
    return EventMatrix(*args, **kwargs)


def event_term(*events, **kwargs):
    """Construct an event term from one or more events."""
    from .events import EventTerm
    if len(events) == 1 and isinstance(events[0], (list, tuple)):
        event_list = list(events[0])
    else:
        event_list = list(events)
    return EventTerm(event_list, **kwargs)


def fmri_dataset(*args, **kwargs):
    """Construct an FmriDataset from an adapter + optional event table."""
    from .dataset import FmriDataset
    return FmriDataset(*args, **kwargs)


def matrix_dataset(
    data,
    tr,
    run_length=None,
    *,
    event_table=None,
    mask=None,
):
    """Construct an in-memory FmriDataset from numpy matrix data.

    Parameters
    ----------
    data : array-like or list[array-like]
        Either a single `(time, voxels)` matrix or a list of run matrices.
    tr : float or list[float]
        Repetition time(s).
    run_length : int or list[int], optional
        If `data` is a single matrix, split into runs by these lengths.
    event_table : pandas.DataFrame, optional
        Event table attached to the resulting dataset.
    mask : ndarray, optional
        Spatial mask passed to the numpy adapter.
    """
    import numpy as np

    from .sampling import SamplingFrame
    from .dataset.adapters import NumpyAdapter
    from .dataset import FmriDataset

    if isinstance(data, list):
        runs = [np.asarray(x, dtype=np.float64) for x in data]
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("matrix_dataset expects 2-D matrix data")
        if run_length is None:
            runs = [arr]
        else:
            if isinstance(run_length, int):
                if run_length <= 0 or (arr.shape[0] % run_length) != 0:
                    raise ValueError(
                        "run_length must divide number of timepoints for matrix input"
                    )
                n_runs = arr.shape[0] // run_length
                blocklens = [run_length] * n_runs
            else:
                blocklens = [int(x) for x in run_length]
                if any(x <= 0 for x in blocklens):
                    raise ValueError("run_length values must be positive")
                if sum(blocklens) != arr.shape[0]:
                    raise ValueError(
                        "sum(run_length) must equal number of timepoints for matrix input"
                    )
            splits = np.cumsum(blocklens)[:-1]
            runs = [np.asarray(x, dtype=np.float64) for x in np.split(arr, splits, axis=0)]

    blocklens = [int(r.shape[0]) for r in runs]
    sf = SamplingFrame(blocklens=blocklens, tr=tr)
    adapter = NumpyAdapter(runs, sf, mask=mask)
    return FmriDataset(adapter, event_table=event_table)

# ── Volume quality helpers (lazy) ──────────────────────────────────
def compute_dvars(*args, **kwargs):
    """Compute DVARS. See :func:`fmrimod.glm.preprocess.compute_dvars`."""
    from .glm.preprocess import compute_dvars as _compute_dvars
    return _compute_dvars(*args, **kwargs)


def dvars_to_weights(*args, **kwargs):
    """Convert DVARS to weights. See :func:`fmrimod.glm.preprocess.dvars_to_weights`."""
    from .glm.preprocess import dvars_to_weights as _dvars_to_weights
    return _dvars_to_weights(*args, **kwargs)


def volume_weights(*args, **kwargs):
    """One-step DVARS-to-weights helper. See :func:`fmrimod.glm.preprocess.volume_weights`."""
    from .glm.preprocess import volume_weights as _volume_weights
    return _volume_weights(*args, **kwargs)

# ── Contrast ─────────────────────────────────────────────────────────
def contrast(*args, **kwargs):
    """Create a contrast. See :func:`fmrimod.contrast.contrast_spec.contrast`."""
    from .contrast.contrast_spec import contrast as _contrast
    return _contrast(*args, **kwargs)


def generate_main_effect_contrast(*args, **kwargs):
    """Generate main-effect F-contrast matrix for a factorial design."""
    from .contrast import generate_main_effect_contrast as _generate_main_effect_contrast
    return _generate_main_effect_contrast(*args, **kwargs)


def generate_interaction_contrast(*args, **kwargs):
    """Generate interaction F-contrast matrix for selected factorial terms."""
    from .contrast import generate_interaction_contrast as _generate_interaction_contrast
    return _generate_interaction_contrast(*args, **kwargs)

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

def soft_subspace_options(*args, **kwargs):
    """Create soft-subspace options for GLM fitting."""
    from .model import soft_subspace_options as _soft_subspace_options
    return _soft_subspace_options(*args, **kwargs)

# ── Beta extraction (lazy) ────────────────────────────────────────
def estimate_betas(*args, **kwargs):
    """Estimate trial-wise betas. See :func:`fmrimod.betas.extraction.estimate_betas`."""
    from .betas.extraction import estimate_betas as _estimate_betas
    return _estimate_betas(*args, **kwargs)

def glm_ols(*args, **kwargs):
    """Estimate trial-wise OLS betas. See :func:`fmrimod.betas.extraction.estimate_betas_ols`."""
    if "progress" in kwargs:
        import warnings

        warnings.warn(
            "'progress' is deprecated and ignored in Python glm_ols wrapper.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.pop("progress", None)
    from .betas.extraction import estimate_betas_ols as _glm_ols
    return _glm_ols(*args, **kwargs)

def glm_lss(*args, **kwargs):
    """Estimate trial-wise LSS betas. See :func:`fmrimod.betas.extraction.estimate_betas_lss`."""
    if "progress" in kwargs:
        import warnings

        warnings.warn(
            "'progress' is deprecated and ignored in Python glm_lss wrapper.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.pop("progress", None)
    if "use_cpp" in kwargs:
        import warnings

        warnings.warn(
            "'use_cpp' is deprecated and ignored; vectorized LSS is always used.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.pop("use_cpp", None)
    from .betas.extraction import estimate_betas_lss as _glm_lss
    return _glm_lss(*args, **kwargs)

def build_nuisance_projector(*args, **kwargs):
    """Build reusable nuisance projector for repeated LSS calls."""
    from .single import build_nuisance_projector as _build_nuisance_projector
    return _build_nuisance_projector(*args, **kwargs)

# ── Bootstrap (lazy) ──────────────────────────────────────────────
def bootstrap_glm(*args, **kwargs):
    """Bootstrap CIs for GLM. See :func:`fmrimod.glm.bootstrap.bootstrap_glm`."""
    from .glm.bootstrap import bootstrap_glm as _bootstrap_glm
    return _bootstrap_glm(*args, **kwargs)


# ── Group data constructors (parity API) ────────────────────────
def group_data(*args, **kwargs):
    """Build a group-level data container for meta-analysis workflows.

    See :func:`fmrimod.dataset.group_data`.
    """
    from .dataset import group_data as _group_data
    return _group_data(*args, **kwargs)


def group_data_from_h5(*args, **kwargs):
    """Construct group data from H5/HDF5 paths."""
    from .dataset import group_data_from_h5 as _group_data_from_h5
    return _group_data_from_h5(*args, **kwargs)


def group_data_from_nifti(*args, **kwargs):
    """Construct group data from NIfTI stats paths."""
    from .dataset import group_data_from_nifti as _group_data_from_nifti
    return _group_data_from_nifti(*args, **kwargs)


def group_data_from_csv(*args, **kwargs):
    """Construct group data from long-format CSV/dataframe inputs."""
    from .dataset import group_data_from_csv as _group_data_from_csv
    return _group_data_from_csv(*args, **kwargs)


def group_data_from_fmrilm(*args, **kwargs):
    """Construct group data from fmri_lm-like fit objects."""
    from .dataset import group_data_from_fmrilm as _group_data_from_fmrilm
    return _group_data_from_fmrilm(*args, **kwargs)


def detect_group_data_format(*args, **kwargs):
    """Detect group-data input format for parity-style auto dispatch."""
    from .dataset import detect_group_data_format as _detect_group_data_format
    return _detect_group_data_format(*args, **kwargs)


# ── Group-level inference (parity API) ────────────────────────────
def fmri_meta(*args, **kwargs):
    """Fit group-level meta-regression.

    See :func:`fmrimod.stats.fmri_meta`.
    """
    from .stats import fmri_meta as _fmri_meta
    return _fmri_meta(*args, **kwargs)


def fmri_ttest(*args, **kwargs):
    """Run a group-level t-test wrapper.

    See :func:`fmrimod.stats.fmri_ttest`.
    """
    from .stats import fmri_ttest as _fmri_ttest
    return _fmri_ttest(*args, **kwargs)


# ── Output helpers ─────────────────────────────────────────────────
def write_results(*args, **kwargs):
    """Write fitted results using BIDS-style filenames.

    See :func:`fmrimod.io.write_results`.
    """
    from .io import write_results as _write_results
    return _write_results(*args, **kwargs)


def t_to_d(*args, **kwargs):
    """Convert t-statistics to standardized effect sizes."""
    from .stats import t_to_d as _t_to_d
    return _t_to_d(*args, **kwargs)


def r_to_z(*args, **kwargs):
    """Convert correlations to Fisher Z and variance."""
    from .stats import r_to_z as _r_to_z
    return _r_to_z(*args, **kwargs)


def z_to_r(*args, **kwargs):
    """Back-transform Fisher Z values to correlations."""
    from .stats import z_to_r as _z_to_r
    return _z_to_r(*args, **kwargs)


def simulate_simple_dataset(*args, **kwargs):
    """Simulate a simple multi-condition dataset with clean/noisy signals."""
    from .simulate import simulate_simple_dataset as _simulate_simple_dataset
    return _simulate_simple_dataset(*args, **kwargs)


def simulate_bold_signal(*args, **kwargs):
    """Simulate condition-wise BOLD responses (compatibility helper)."""
    from .simulate import simulate_bold_signal as _simulate_bold_signal
    return _simulate_bold_signal(*args, **kwargs)


def simulate_noise_vector(*args, **kwargs):
    """Simulate fMRI-like 1D noise with ARMA/drift/physiology components."""
    from .simulate import simulate_noise_vector as _simulate_noise_vector
    return _simulate_noise_vector(*args, **kwargs)


def simulate_fmri_matrix(*args, **kwargs):
    """Simulate a matrix_dataset-like object with event/noise metadata."""
    from .simulate import simulate_fmri_matrix as _simulate_fmri_matrix
    return _simulate_fmri_matrix(*args, **kwargs)


def estimate_hrf(*args, **kwargs):
    """Estimate HRFs from matrix inputs (initial parity workflow)."""
    from .single import estimate_hrf as _estimate_hrf
    return _estimate_hrf(*args, **kwargs)


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
    "event_factor",
    "event_variable",
    "event_matrix",
    "event_term",
    "fmri_dataset",
    "matrix_dataset",
    "compute_dvars",
    "dvars_to_weights",
    "volume_weights",
    "contrast",
    "generate_main_effect_contrast",
    "generate_interaction_contrast",
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
    "soft_subspace_options",
    # Beta extraction
    "estimate_betas",
    "glm_ols",
    "glm_lss",
    "build_nuisance_projector",
    # Bootstrap
    "bootstrap_glm",
    # Group data constructors
    "group_data",
    "group_data_from_h5",
    "group_data_from_nifti",
    "group_data_from_csv",
    "group_data_from_fmrilm",
    "detect_group_data_format",
    # Group-level inference
    "fmri_meta",
    "fmri_ttest",
    # Output helpers
    "write_results",
    "t_to_d",
    "r_to_z",
    "z_to_r",
    "simulate_simple_dataset",
    "simulate_bold_signal",
    "simulate_noise_vector",
    "simulate_fmri_matrix",
    "estimate_hrf",
]
