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
from .hrf.functions import spm_canonical, gamma_hrf, gaussian_hrf
from .hrf.registry import get_hrf, list_available_hrfs

# ── HRF generators ───────────────────────────────────────────────────
from .hrf.generators import (
    gen_hrf,
    gen_hrf_set,
    hrf_set,
    hrf_bspline_generator,
    hrf_fir_generator,
    hrf_fourier_generator,
    hrf_daguerre_generator,
    hrf_tent_generator,
)
from .hrf.decorators import lag_hrf, block_hrf, hrf_lagged, hrf_blocked
from .hrf.empirical import gen_empirical_hrf
from .hrf.hrf_library import gen_hrf_library

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


def fmri_mem_dataset(*args, **kwargs):
    """Construct an in-memory fMRI dataset."""
    from .dataset import fmri_mem_dataset as _fmri_mem_dataset
    return _fmri_mem_dataset(*args, **kwargs)


def latent_dataset(*args, **kwargs):
    """Construct a latent-component dataset."""
    from .dataset import latent_dataset as _latent_dataset
    return _latent_dataset(*args, **kwargs)


def fmri_latent_lm(*args, **kwargs):
    """Fit a GLM to latent-component scores."""
    from .dataset import fmri_latent_lm as _fmri_latent_lm
    return _fmri_latent_lm(*args, **kwargs)


def data_chunks(*args, **kwargs):
    """Return voxel-index chunks for a dataset or array."""
    from .dataset import data_chunks as _data_chunks
    return _data_chunks(*args, **kwargs)


def extract_csv_data(*args, **kwargs):
    """Extract effect-size arrays from CSV-backed group data."""
    from .dataset import extract_csv_data as _extract_csv_data
    return _extract_csv_data(*args, **kwargs)


def read_h5_full(*args, **kwargs):
    """Read full HDF5-backed group data."""
    from .dataset import read_h5_full as _read_h5_full
    return _read_h5_full(*args, **kwargs)


def read_nifti_full(*args, **kwargs):
    """Read full NIfTI-backed group data."""
    from .dataset import read_nifti_full as _read_nifti_full
    return _read_nifti_full(*args, **kwargs)


def read_fmri_config(*args, **kwargs):
    """Read a JSON or YAML fMRI config file."""
    from .dataset import read_fmri_config as _read_fmri_config
    return _read_fmri_config(*args, **kwargs)


def register_basis(*args, **kwargs):
    """Register a custom basis constructor."""
    from .dataset import register_basis as _register_basis
    return _register_basis(*args, **kwargs)


def resolve_basis(*args, **kwargs):
    """Resolve a registered or built-in basis by name."""
    from .dataset import resolve_basis as _resolve_basis
    return _resolve_basis(*args, **kwargs)


def load_benchmark_dataset(*args, **kwargs):
    """Load a deterministic built-in benchmark fixture."""
    from .dataset import load_benchmark_dataset as _load_benchmark_dataset
    return _load_benchmark_dataset(*args, **kwargs)


def list_benchmark_datasets(*args, **kwargs):
    """List deterministic built-in benchmark fixtures."""
    from .dataset import list_benchmark_datasets as _list_benchmark_datasets
    return _list_benchmark_datasets(*args, **kwargs)


def get_benchmark_summary(*args, **kwargs):
    """Return summary metadata for a benchmark fixture."""
    from .dataset import get_benchmark_summary as _get_benchmark_summary
    return _get_benchmark_summary(*args, **kwargs)


def create_design_matrix_from_benchmark(*args, **kwargs):
    """Create a design matrix for a benchmark fixture."""
    from .dataset import create_design_matrix_from_benchmark as _create_design_matrix_from_benchmark
    return _create_design_matrix_from_benchmark(*args, **kwargs)


def evaluate_method_performance(*args, **kwargs):
    """Evaluate estimated betas against a benchmark fixture."""
    from .dataset import evaluate_method_performance as _evaluate_method_performance
    return _evaluate_method_performance(*args, **kwargs)


def design_plot(*args, **kwargs):
    """Return a long-form design matrix table suitable for plotting."""
    from .dataset import design_plot as _design_plot
    return _design_plot(*args, **kwargs)

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


def contrast_weights(*args, **kwargs):
    """Compute contrast weights. See :func:`fmrimod.contrast.contrast_weights`."""
    from .contrast import contrast_weights as _contrast_weights
    return _contrast_weights(*args, **kwargs)


def Fcontrasts(*args, **kwargs):
    """Create F-contrasts. See :func:`fmrimod.contrast.Fcontrasts`."""
    from .contrast import Fcontrasts as _Fcontrasts
    return _Fcontrasts(*args, **kwargs)


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

def condition_map(*args, **kwargs):
    """Map display and canonical condition names."""
    from .utils import condition_map as _condition_map
    return _condition_map(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Evaluate an HRF, basis, regressor, or callable object."""
    from .utils import evaluate as _evaluate
    return _evaluate(*args, **kwargs)


def acquisition_onsets(*args, **kwargs):
    """Return global fMRI acquisition onset times."""
    from .utils import acquisition_onsets as _acquisition_onsets
    return _acquisition_onsets(*args, **kwargs)


def amplitudes(*args, **kwargs):
    """Return event amplitudes from a regressor-like object."""
    from .utils import amplitudes as _amplitudes
    return _amplitudes(*args, **kwargs)


def samples(*args, **kwargs):
    """Return sampling times from a sampling-frame-like object."""
    from .utils import samples as _samples
    return _samples(*args, **kwargs)


def global_onsets(*args, **kwargs):
    """Convert block-local event onsets to global experiment onsets."""
    from .utils import global_onsets as _global_onsets
    return _global_onsets(*args, **kwargs)


def shift(*args, **kwargs):
    """Shift a regressor-like object by a temporal offset."""
    from .utils import shift as _shift
    return _shift(*args, **kwargs)


def ar_parameters(*args, **kwargs):
    """Return AR parameters from a fitted GLM result."""
    from .accessors import ar_parameters as _ar_parameters
    return _ar_parameters(*args, **kwargs)


def coef_image(*args, **kwargs):
    """Return coefficient/statistic values, optionally reconstructed into a mask."""
    from .accessors import coef_image as _coef_image
    return _coef_image(*args, **kwargs)


def coef_names(*args, **kwargs):
    """Return coefficient names from a model or fitted GLM result."""
    from .accessors import coef_names as _coef_names
    return _coef_names(*args, **kwargs)


def fitted_hrf(*args, **kwargs):
    """Return fitted HRF reconstructions for HRF-coded event terms."""
    from .accessors import fitted_hrf as _fitted_hrf
    return _fitted_hrf(*args, **kwargs)


def get_contrasts(*args, **kwargs):
    """Return contrast names/specifications from fitted or group-data objects."""
    from .accessors import get_contrasts as _get_contrasts
    return _get_contrasts(*args, **kwargs)


def get_covariates(*args, **kwargs):
    """Return covariates from group-data objects."""
    from .accessors import get_covariates as _get_covariates
    return _get_covariates(*args, **kwargs)


def get_data(*args, **kwargs):
    """Return run data from a dataset-like object."""
    from .accessors import get_data as _get_data
    return _get_data(*args, **kwargs)


def get_data_matrix(*args, **kwargs):
    """Return all run data concatenated along time."""
    from .accessors import get_data_matrix as _get_data_matrix
    return _get_data_matrix(*args, **kwargs)


def get_formula(*args, **kwargs):
    """Return the stored formula or a term summary when available."""
    from .accessors import get_formula as _get_formula
    return _get_formula(*args, **kwargs)


def get_mask(*args, **kwargs):
    """Return a boolean mask from a dataset-like object."""
    from .accessors import get_mask as _get_mask
    return _get_mask(*args, **kwargs)


def get_rois(*args, **kwargs):
    """Return ROI labels from group-data objects."""
    from .accessors import get_rois as _get_rois
    return _get_rois(*args, **kwargs)


def get_subjects(*args, **kwargs):
    """Return subject identifiers from group-data objects."""
    from .accessors import get_subjects as _get_subjects
    return _get_subjects(*args, **kwargs)


def n_subjects(*args, **kwargs):
    """Return the number of subjects in a group-data object."""
    from .accessors import n_subjects as _n_subjects
    return _n_subjects(*args, **kwargs)


def p_values(*args, **kwargs):
    """Return p-values from a fitted GLM result."""
    from .accessors import p_values as _p_values
    return _p_values(*args, **kwargs)


def pvalues(*args, **kwargs):
    """Alias for :func:`p_values`."""
    from .accessors import pvalues as _pvalues
    return _pvalues(*args, **kwargs)


def se(*args, **kwargs):
    """Alias for :func:`standard_error`."""
    from .accessors import se as _se
    return _se(*args, **kwargs)


def standard_error(*args, **kwargs):
    """Return standard errors from a fitted GLM result."""
    from .accessors import standard_error as _standard_error
    return _standard_error(*args, **kwargs)


def stats(*args, **kwargs):
    """Return coefficient or contrast statistics from a fitted GLM result."""
    from .accessors import stats as _stats
    return _stats(*args, **kwargs)


def tidy(*args, **kwargs):
    """Return fitted-model information as a tidy DataFrame."""
    from .accessors import tidy as _tidy
    return _tidy(*args, **kwargs)


def tidy_fitted_hrf(*args, **kwargs):
    """Return fitted HRF reconstructions as a tidy DataFrame."""
    from .accessors import tidy_fitted_hrf as _tidy_fitted_hrf
    return _tidy_fitted_hrf(*args, **kwargs)


def zscores(*args, **kwargs):
    """Return z-score equivalents for fitted statistics."""
    from .accessors import zscores as _zscores
    return _zscores(*args, **kwargs)


# ── AR/ARMA noise whitening (fmriAR parity API) ────────────────────
def fit_noise(*args, **kwargs):
    """Fit an AR/ARMA noise model and return a whitening plan."""
    from .ar import fit_noise as _fit_noise
    return _fit_noise(*args, **kwargs)


def whiten_apply(*args, **kwargs):
    """Apply a fitted whitening plan to design and data matrices."""
    from .ar import whiten_apply as _whiten_apply
    return _whiten_apply(*args, **kwargs)


def whiten(*args, **kwargs):
    """Fit and apply AR/ARMA whitening in one call."""
    from .ar import whiten as _whiten
    return _whiten(*args, **kwargs)


def acorr_diagnostics(*args, **kwargs):
    """Compute residual autocorrelation diagnostics."""
    from .ar import acorr_diagnostics as _acorr_diagnostics
    return _acorr_diagnostics(*args, **kwargs)


def sandwich_from_whitened_resid(*args, **kwargs):
    """Compute GLS standard errors from whitened residuals."""
    from .ar import sandwich_from_whitened_resid as _sandwich_from_whitened_resid
    return _sandwich_from_whitened_resid(*args, **kwargs)


def afni_restricted_plan(*args, **kwargs):
    """Build an AFNI-style restricted AR whitening plan."""
    from .ar import afni_restricted_plan as _afni_restricted_plan
    return _afni_restricted_plan(*args, **kwargs)


from .ar import compat

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


def soft_projection(*args, **kwargs):
    """Create a ridge-regularized nuisance projection."""
    from .glm import soft_projection as _soft_projection
    return _soft_projection(*args, **kwargs)


def apply_soft_projection(*args, **kwargs):
    """Apply a soft projection to response and design matrices."""
    from .glm import apply_soft_projection as _apply_soft_projection
    return _apply_soft_projection(*args, **kwargs)


def compute_lm_contrasts(*args, **kwargs):
    """Compute t/F contrast statistics from fitted GLM matrices."""
    from .glm import compute_lm_contrasts as _compute_lm_contrasts
    return _compute_lm_contrasts(*args, **kwargs)


def compute_lm_contrasts_from_suffstats(*args, **kwargs):
    """Compute contrast statistics from design/data sufficient statistics."""
    from .glm import compute_lm_contrasts_from_suffstats as _compute_lm_contrasts_from_suffstats
    return _compute_lm_contrasts_from_suffstats(*args, **kwargs)


def fit_contrasts(*args, **kwargs):
    """Fit contrasts on a fitted GLM result or matrix payload."""
    from .glm import fit_contrasts as _fit_contrasts
    return _fit_contrasts(*args, **kwargs)


def fit_glm_on_transformed_series(*args, **kwargs):
    """Fit a GLM with an externally supplied response matrix."""
    from .glm import fit_glm_on_transformed_series as _fit_glm_on_transformed_series
    return _fit_glm_on_transformed_series(*args, **kwargs)


def fit_glm_with_config(*args, **kwargs):
    """Fit an externally supplied response matrix with an FmriLmConfig."""
    from .glm import fit_glm_with_config as _fit_glm_with_config
    return _fit_glm_with_config(*args, **kwargs)


def fit_glm_from_suffstats(*args, **kwargs):
    """Build a GLM result from sufficient statistics."""
    from .glm import fit_glm_from_suffstats as _fit_glm_from_suffstats
    return _fit_glm_from_suffstats(*args, **kwargs)


def fmri_ols_fit(*args, **kwargs):
    """Fit matrix OLS and return beta, SE, and t summaries."""
    from .glm import fmri_ols_fit as _fmri_ols_fit
    return _fmri_ols_fit(*args, **kwargs)


def fmri_rlm(*args, **kwargs):
    """Fit a robust GLM using the Python model contract."""
    from .glm import fmri_rlm as _fmri_rlm
    return _fmri_rlm(*args, **kwargs)


def lowrank_control(*args, **kwargs):
    """Create low-rank/sketch fitting options."""
    from .glm import lowrank_control as _lowrank_control
    return _lowrank_control(*args, **kwargs)


def paired_diff_block(*args, **kwargs):
    """Compute paired within-subject differences for group-data blocks."""
    from .glm import paired_diff_block as _paired_diff_block
    return _paired_diff_block(*args, **kwargs)


def flip_sign(*args, **kwargs):
    """Flip coefficient-like signed outputs in a mapping or fitted object."""
    from .glm import flip_sign as _flip_sign
    return _flip_sign(*args, **kwargs)


def t_to_beta_se(*args, **kwargs):
    """Convert t-statistics to approximate beta and SE estimates."""
    from .glm import t_to_beta_se as _t_to_beta_se
    return _t_to_beta_se(*args, **kwargs)


def hrf_smoothing_kernel(*args, **kwargs):
    """Compute a temporal smoothing kernel from a trialwise design matrix."""
    from .glm import hrf_smoothing_kernel as _hrf_smoothing_kernel
    return _hrf_smoothing_kernel(*args, **kwargs)


def estimate(*args, **kwargs):
    """Deprecated compatibility helper; use estimate_betas instead."""
    from .glm import estimate as _estimate
    return _estimate(*args, **kwargs)

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


def fmri_meta_fit(*args, **kwargs):
    """Fit low-level matrix meta-regression."""
    from .stats import fmri_meta_fit as _fmri_meta_fit
    return _fmri_meta_fit(*args, **kwargs)


def fmri_meta_fit_contrasts(*args, **kwargs):
    """Fit low-level matrix meta-regression with exact contrasts."""
    from .stats import fmri_meta_fit_contrasts as _fmri_meta_fit_contrasts
    return _fmri_meta_fit_contrasts(*args, **kwargs)


def fmri_meta_fit_cov(*args, **kwargs):
    """Fit low-level matrix meta-regression with packed covariance output."""
    from .stats import fmri_meta_fit_cov as _fmri_meta_fit_cov
    return _fmri_meta_fit_cov(*args, **kwargs)


def fmri_meta_fit_extended(*args, **kwargs):
    """Fit low-level matrix meta-regression with optional voxelwise covariates."""
    from .stats import fmri_meta_fit_extended as _fmri_meta_fit_extended
    return _fmri_meta_fit_extended(*args, **kwargs)


def meta_effective_n(*args, **kwargs):
    """Compute inverse-variance effective sample size."""
    from .stats import meta_effective_n as _meta_effective_n
    return _meta_effective_n(*args, **kwargs)


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
    "spm_canonical",
    "gamma_hrf",
    "gaussian_hrf",
    # HRF registry
    "get_hrf",
    "list_available_hrfs",
    # HRF generators & decorators
    "gen_hrf",
    "gen_hrf_set",
    "hrf_set",
    "gen_empirical_hrf",
    "gen_hrf_library",
    "lag_hrf",
    "block_hrf",
    "hrf_lagged",
    "hrf_blocked",
    "hrf_bspline_generator",
    "hrf_fir_generator",
    "hrf_fourier_generator",
    "hrf_daguerre_generator",
    "hrf_tent_generator",
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
    "fmri_mem_dataset",
    "latent_dataset",
    "fmri_latent_lm",
    "data_chunks",
    "extract_csv_data",
    "read_h5_full",
    "read_nifti_full",
    "read_fmri_config",
    "register_basis",
    "resolve_basis",
    "load_benchmark_dataset",
    "list_benchmark_datasets",
    "get_benchmark_summary",
    "create_design_matrix_from_benchmark",
    "evaluate_method_performance",
    "design_plot",
    "compute_dvars",
    "dvars_to_weights",
    "volume_weights",
    "contrast",
    "contrast_weights",
    "Fcontrasts",
    "generate_main_effect_contrast",
    "generate_interaction_contrast",
    # Basis functions
    "Poly",
    "Scale",
    "RobustScale",
    "condition_basis_list",
    "condition_map",
    "evaluate",
    "acquisition_onsets",
    "amplitudes",
    "samples",
    "global_onsets",
    "shift",
    "ar_parameters",
    "coef_image",
    "coef_names",
    "fitted_hrf",
    "get_contrasts",
    "get_covariates",
    "get_data",
    "get_data_matrix",
    "get_formula",
    "get_mask",
    "get_rois",
    "get_subjects",
    "n_subjects",
    "p_values",
    "pvalues",
    "se",
    "standard_error",
    "stats",
    "tidy",
    "tidy_fitted_hrf",
    "zscores",
    "fit_noise",
    "whiten",
    "whiten_apply",
    "acorr_diagnostics",
    "sandwich_from_whitened_resid",
    "afni_restricted_plan",
    "compat",
    # Formula functions
    "hrf_formula",
    "hrf_spmg1",
    # GLM fitting
    "fmri_lm",
    "soft_subspace_options",
    "soft_projection",
    "apply_soft_projection",
    "compute_lm_contrasts",
    "compute_lm_contrasts_from_suffstats",
    "fit_contrasts",
    "fit_glm_on_transformed_series",
    "fit_glm_with_config",
    "fit_glm_from_suffstats",
    "fmri_ols_fit",
    "fmri_rlm",
    "lowrank_control",
    "paired_diff_block",
    "flip_sign",
    "t_to_beta_se",
    "hrf_smoothing_kernel",
    "estimate",
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
    "fmri_meta_fit",
    "fmri_meta_fit_contrasts",
    "fmri_meta_fit_cov",
    "fmri_meta_fit_extended",
    "meta_effective_n",
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
