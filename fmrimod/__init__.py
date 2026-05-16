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

import inspect as _inspect
from typing import Any, Callable, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ── Core classes ──────────────────────────────────────────────────────
# Force-load the callable stats subpackage so `fmrimod.stats` is bound to
# the `_CallableStatsModule` instance regardless of test/import order. The
# subpackage's __init__ installs the callable protocol on itself; without
# this eager import, `fmrimod.stats` would briefly resolve to the accessor
# function (via __getattr__) and then be silently overwritten the first
# time anything does `from fmrimod.stats import ...`.
from . import stats as stats  # noqa: F401  (eager binding, used via attribute)
from .base import BaseEvent
from .betas.extraction import BetaResult
from .contrast import (
    column_contrast as column_contrast,
)
from .contrast import (
    contrast_set as contrast_set,
)
from .contrast import (
    interaction_contrast as interaction_contrast,
)
from .contrast import (
    one_against_all_contrast as one_against_all_contrast,
)
from .contrast import (
    oneway_contrast as oneway_contrast,
)

# Namespace shadowing — ``hrf``/``trialwise`` are withheld here and
# ``contrast``/``regressor`` rebind their submodules. The *rule* and the
# enumerated allowed set live in exactly one home per GOVERNANCE G8:
# docs/contracts/public_api_policy_v1.md § "Namespace shadowing", pinned
# by tests/test_public_api/test_namespace_shadowing.py (the ``contrast``
# dual-resolution is additionally anchored by
# tests/test_contrast/test_polymorphic_predicates.py). Do not restate the
# rule here; this comment is only a pointer.
# ── Contrast taxonomy (bd-01KRFMD3F66TENJMP6BQYE32HC) ──────────────────
from .contrast import (
    pair_contrast as pair_contrast,
)
from .contrast import (
    pairwise_contrasts as pairwise_contrasts,
)
from .contrast import (
    poly_contrast as poly_contrast,
)
from .contrast import (
    sliding_window_contrasts as sliding_window_contrasts,
)
from .contrast import (
    unit_contrast as unit_contrast,
)
from .contrast.contrast_spec import contrast as contrast
from .dataset.fmri_dataset import FmriDataset
from .events.term import EventTerm
from .formula.base import Term as FormulaTerm
from .hrf.core import HRF
from .hrf.decorators import block_hrf, hrf_blocked, hrf_lagged, lag_hrf
from .hrf.empirical import gen_empirical_hrf
from .hrf.functions import gamma_hrf, gaussian_hrf, spm_canonical

# ── HRF generators ───────────────────────────────────────────────────
from .hrf.generators import (
    gen_hrf,
    gen_hrf_set,
    hrf_bspline_generator,
    hrf_daguerre_generator,
    hrf_fir_generator,
    hrf_fourier_generator,
    hrf_set,
    hrf_tent_generator,
)
from .hrf.hrf_library import gen_hrf_library

# ── Pre-defined HRFs (most commonly used) ────────────────────────────
from .hrf.library import (
    HRF_BSPLINE,
    HRF_FIR,
    HRF_FOURIER,
    HRF_GAMMA,
    HRF_GAUSSIAN,
    HRF_HALF_COSINE,
    HRF_INV_LOGIT,
    HRF_LWU,
    HRF_LWU_BASIS,
    HRF_MEXHAT,
    HRF_SINE,
    # R-parity aliases — match the fmrihrf constant names for ported code
    HRF_SPMG1,
    HRF_SPMG2,
    HRF_SPMG3,
    HRF_TIME,
    SPM_CANONICAL,
    SPM_WITH_DERIVATIVE,
    SPM_WITH_DISPERSION,
)
from .hrf.registry import get_hrf, list_available_hrfs
from .hrf_dispatch import as_hrf
from .regressor import null_regressor, regressor, regressor_set
from .sampling import SamplingFrame

# ── Typed Spec / Term tree (bd-01KRFMD3CXMEMHZXBKP9T0EAK6) ─────────────
# Bind the spec builders at the top level. The submodule ``fmrimod.hrf``
# remains importable via the Python import system (``from fmrimod.hrf import
# HRF_SPMG1``); only attribute access on the top-level ``fmrimod`` package
# resolves to the spec builder.
from .spec import (
    Confounds,
    Drift,
    FieldDiff,
    HrfTerm,
    Intercept,
    Spec,
    SpecDiff,
    SpecSerializationError,
    Term,
    TermDiff,
    as_spec,
    is_spec,
    spec_diff,
)
from .spec.builders import (
    confounds as confounds,
)
from .spec.builders import (
    drift as drift,
)
from .spec.builders import (
    intercept as intercept,
)
from .types import HRFProtocol

# ── Design (lazy imports to avoid circular dependencies) ─────────────
# ── Baseline ─────────────────────────────────────────────────────────


def event_term(
    event: "Union[BaseEvent, Sequence[BaseEvent]]",
    event2: "Optional[BaseEvent]" = None,
    event3: "Optional[BaseEvent]" = None,
    event4: "Optional[BaseEvent]" = None,
    *,
    name: Optional[str] = None,
    interaction: bool = False,
) -> "EventTerm":
    """Construct an event term from one or more events."""
    from .events import EventTerm

    additional = [item for item in (event2, event3, event4) if item is not None]
    if not additional and isinstance(event, (list, tuple)):
        event_list = list(event)
    else:
        event_list = [event, *additional]
    return EventTerm(event_list, name=name, interaction=interaction)


def matrix_dataset(
    data: "Union[NDArray[np.float64], Sequence[NDArray[np.float64]]]",
    tr: "Union[float, list[float], None]" = None,
    run_length: "Union[int, list[int], None]" = None,
    *,
    event_table: "Optional[pd.DataFrame]" = None,
    mask: "Optional[NDArray[np.bool_]]" = None,
    TR: "Union[float, list[float], None]" = None,
) -> "FmriDataset":
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
    from .dataset import matrix_dataset as _matrix_dataset

    return _matrix_dataset(
        data,
        tr,
        run_length=run_length,
        event_table=event_table,
        mask=mask,
        TR=TR,
    )


# ── Volume quality helpers (lazy) ──────────────────────────────────


# ── Contrast ─────────────────────────────────────────────────────────


# ── Basis functions ──────────────────────────────────────────────────
# ── Formula functions (lazy to avoid shadowing hrf subpackage) ───────
from functools import partial as _partial

# ── AR/ARMA noise whitening (fmriAR parity API) ────────────────────
from .ar import compat
from .basis.polynomial import Poly
from .basis.transform import RobustScale, Scale
from .condition_basis import condition_basis_list


def hrf_formula(
    spec: "Union[str, HRFProtocol]" = "spmg1",
    *,
    subset: object = None,
    contrasts: object = None,
    normalize: bool = False,
    summate: bool = True,
    hrf_fun: Optional[Callable[..., object]] = None,
    id: Optional[str] = None,
    prefix: Optional[str] = None,
    lag: float = 0.0,
    nbasis: int = 1,
    onsets: object = None,
    durations: object = None,
) -> "Callable[[FormulaTerm], FormulaTerm]":
    """HRF formula function. See :func:`fmrimod.formula.functional.hrf`."""
    from .formula.functional import hrf as _hrf_func

    return _hrf_func(
        spec,
        subset=subset,
        contrasts=contrasts,
        normalize=normalize,
        summate=summate,
        hrf_fun=hrf_fun,
        id=id,
        prefix=prefix,
        lag=lag,
        nbasis=nbasis,
        onsets=onsets,
        durations=durations,
    )


hrf_spmg1 = _partial(hrf_formula, spec="spmg1")
cast(Any, hrf_spmg1).__signature__ = _inspect.signature(hrf_formula).replace(
    parameters=[
        param
        for param in _inspect.signature(hrf_formula).parameters.values()
        if param.name != "spec"
    ]
)

# ── GLM fitting (lazy imports) ──────────────────────────────────────


from .glm.spatial import SpatialContext  # noqa: E402


# ── Beta extraction (lazy) ────────────────────────────────────────
def glm_ols(
    trial_regressors: "NDArray[np.float64]",
    Y: "NDArray[np.float64]",
    confounds: "Optional[NDArray[np.float64]]" = None,
    baseline_regressors: "Optional[NDArray[np.float64]]" = None,
    include_intercept: bool = False,
    *,
    progress: Optional[bool] = None,
) -> "BetaResult":
    """Estimate trial-wise OLS betas."""
    if progress is not None:
        import warnings

        warnings.warn(
            "'progress' is deprecated and ignored in Python glm_ols wrapper.",
            DeprecationWarning,
            stacklevel=2,
        )
    from .betas.extraction import estimate_betas_ols as _glm_ols

    return _glm_ols(
        trial_regressors,
        Y,
        confounds=confounds,
        baseline_regressors=baseline_regressors,
        include_intercept=include_intercept,
    )


def glm_lss(
    trial_regressors: "NDArray[np.float64]",
    Y: "NDArray[np.float64]",
    confounds: "Optional[NDArray[np.float64]]" = None,
    nuisance_projector: object = None,
    chunk_size: Optional[int] = None,
    baseline_regressors: "Optional[NDArray[np.float64]]" = None,
    include_intercept: bool = False,
    *,
    progress: Optional[bool] = None,
    use_cpp: Optional[bool] = None,
) -> "BetaResult":
    """Estimate trial-wise LSS betas."""
    if progress is not None:
        import warnings

        warnings.warn(
            "'progress' is deprecated and ignored in Python glm_lss wrapper.",
            DeprecationWarning,
            stacklevel=2,
        )
    if use_cpp is not None:
        import warnings

        warnings.warn(
            "'use_cpp' is deprecated and ignored; vectorized LSS is always used.",
            DeprecationWarning,
            stacklevel=2,
        )
    from .betas.extraction import estimate_betas_lss as _glm_lss

    return _glm_lss(
        trial_regressors,
        Y,
        confounds=confounds,
        nuisance_projector=nuisance_projector,
        chunk_size=chunk_size,
        baseline_regressors=baseline_regressors,
        include_intercept=include_intercept,
    )


# ── Bootstrap (lazy) ──────────────────────────────────────────────

# ── Group data constructors (parity API) ────────────────────────


# ── Group-level inference (parity API) ────────────────────────────


# ── Output helpers ─────────────────────────────────────────────────


# ─ Lazy attribute table ─────────────────────────────────────────────
#
# Replaces 120 boilerplate (*args, **kwargs) re-export wrappers. Each
# entry is `name -> (module_path, attribute_name)`. The first access
# imports the owning submodule and caches the real callable on this
# module, so subsequent lookups are plain attribute accesses and
# `inspect.signature(fmrimod.<name>)` returns the implementation's
# real signature rather than the prior opaque `(*args, **kwargs)`.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Fcontrasts": ("fmrimod.contrast", "Fcontrasts"),
    "acorr_diagnostics": ("fmrimod.ar", "acorr_diagnostics"),
    "acquisition_onsets": ("fmrimod.utils", "acquisition_onsets"),
    "afni_restricted_plan": ("fmrimod.ar", "afni_restricted_plan"),
    "amplitudes": ("fmrimod.utils", "amplitudes"),
    "apply_soft_projection": ("fmrimod.glm", "apply_soft_projection"),
    "ar_parameters": ("fmrimod.accessors", "ar_parameters"),
    "baseline_model": ("fmrimod.baseline", "baseline_model"),
    "bootstrap_glm": ("fmrimod.glm.bootstrap", "bootstrap_glm"),
    "build_nuisance_projector": ("fmrimod.single", "build_nuisance_projector"),
    "check_nuisance": ("fmrimod.baseline", "check_nuisance"),
    "clean_nuisance": ("fmrimod.baseline", "clean_nuisance"),
    "coef_image": ("fmrimod.accessors", "coef_image"),
    "coef_names": ("fmrimod.accessors", "coef_names"),
    "combine_contrasts": ("fmrimod.glm.combine", "combine_contrasts"),
    "combine_runs": ("fmrimod.glm.combine", "combine_runs"),
    "compute_dvars": ("fmrimod.glm.preprocess", "compute_dvars"),
    "compute_lm_contrasts": ("fmrimod.glm", "compute_lm_contrasts"),
    "compute_lm_contrasts_from_suffstats": (
        "fmrimod.glm",
        "compute_lm_contrasts_from_suffstats",
    ),
    "condition_map": ("fmrimod.utils", "condition_map"),
    "ContrastDelta": ("fmrimod.glm", "ContrastDelta"),
    "contrast_weights": ("fmrimod.contrast", "contrast_weights"),
    "create_design_matrix_from_benchmark": (
        "fmrimod.dataset",
        "create_design_matrix_from_benchmark",
    ),
    "data_chunks": ("fmrimod.dataset", "data_chunks"),
    "design_matrix": ("fmrimod.design.design_matrix", "design_matrix"),
    "design_plot": ("fmrimod.dataset", "design_plot"),
    "detect_group_data_format": ("fmrimod.dataset", "detect_group_data_format"),
    "dvars_to_weights": ("fmrimod.glm.preprocess", "dvars_to_weights"),
    "estimate": ("fmrimod.glm", "estimate"),
    "estimate_betas": ("fmrimod.betas.extraction", "estimate_betas"),
    "estimate_hrf": ("fmrimod.single", "estimate_hrf"),
    "estimate_single_trial": ("fmrimod.single", "estimate_single_trial"),
    "estimate_single_trial_from_dataset": (
        "fmrimod.single",
        "estimate_single_trial_from_dataset",
    ),
    "evaluate": ("fmrimod.utils", "evaluate"),
    "evaluate_method_performance": ("fmrimod.dataset", "evaluate_method_performance"),
    "event_factor": ("fmrimod.events", "EventFactor"),
    "event_matrix": ("fmrimod.events", "EventMatrix"),
    "event_model": ("fmrimod.design.event_model", "event_model"),
    "event_variable": ("fmrimod.events", "EventVariable"),
    "extract_csv_data": ("fmrimod.dataset", "extract_csv_data"),
    "fit_contrasts": ("fmrimod.glm", "fit_contrasts"),
    "fit_glm_from_matrix": ("fmrimod.glm", "fit_glm_from_matrix"),
    "fit_glm_from_suffstats": ("fmrimod.glm", "fit_glm_from_suffstats"),
    "fit_glm_on_transformed_series": ("fmrimod.glm", "fit_glm_on_transformed_series"),
    "fit_glm_with_config": ("fmrimod.glm", "fit_glm_with_config"),
    "fit_noise": ("fmrimod.ar", "fit_noise"),
    "fitted_hrf": ("fmrimod.accessors", "fitted_hrf"),
    "flip_sign": ("fmrimod.glm", "flip_sign"),
    "fmri_dataset": ("fmrimod.dataset", "fmri_dataset"),
    "fmri_latent_lm": ("fmrimod.dataset", "fmri_latent_lm"),
    "fmri_lm": ("fmrimod.glm.fmri_lm", "fmri_lm"),
    "fmri_mem_dataset": ("fmrimod.dataset", "fmri_mem_dataset"),
    "fmri_meta": ("fmrimod.stats", "fmri_meta"),
    "fmri_meta_fit": ("fmrimod.stats", "fmri_meta_fit"),
    "fmri_meta_fit_contrasts": ("fmrimod.stats", "fmri_meta_fit_contrasts"),
    "fmri_meta_fit_cov": ("fmrimod.stats", "fmri_meta_fit_cov"),
    "fmri_meta_fit_extended": ("fmrimod.stats", "fmri_meta_fit_extended"),
    "fmri_ols_fit": ("fmrimod.glm", "fmri_ols_fit"),
    "fmri_rlm": ("fmrimod.glm", "fmri_rlm"),
    "fmri_ttest": ("fmrimod.stats", "fmri_ttest"),
    "fmrihrf_cli": ("fmrimod.cli", "fmrihrf_cli"),
    "generate_interaction_contrast": (
        "fmrimod.contrast",
        "generate_interaction_contrast",
    ),
    "generate_main_effect_contrast": (
        "fmrimod.contrast",
        "generate_main_effect_contrast",
    ),
    "get_benchmark_summary": ("fmrimod.dataset", "get_benchmark_summary"),
    "get_contrasts": ("fmrimod.accessors", "get_contrasts"),
    "get_covariates": ("fmrimod.accessors", "get_covariates"),
    "get_data": ("fmrimod.accessors", "get_data"),
    "get_data_matrix": ("fmrimod.accessors", "get_data_matrix"),
    "get_formula": ("fmrimod.accessors", "get_formula"),
    "get_mask": ("fmrimod.accessors", "get_mask"),
    "get_rois": ("fmrimod.accessors", "get_rois"),
    "get_subjects": ("fmrimod.accessors", "get_subjects"),
    "global_onsets": ("fmrimod.utils", "global_onsets"),
    "group_data": ("fmrimod.dataset", "group_data"),
    "group_data_from_csv": ("fmrimod.dataset", "group_data_from_csv"),
    "group_data_from_fmrilm": ("fmrimod.dataset", "group_data_from_fmrilm"),
    "group_data_from_h5": ("fmrimod.dataset", "group_data_from_h5"),
    "group_data_from_nifti": ("fmrimod.dataset", "group_data_from_nifti"),
    "hrf_smoothing_kernel": ("fmrimod.glm", "hrf_smoothing_kernel"),
    "install_cli": ("fmrimod.cli", "install_cli"),
    "latent_dataset": ("fmrimod.dataset", "latent_dataset"),
    "list_benchmark_datasets": ("fmrimod.dataset", "list_benchmark_datasets"),
    "load_benchmark_dataset": ("fmrimod.dataset", "load_benchmark_dataset"),
    "lowrank_control": ("fmrimod.glm", "lowrank_control"),
    "lsa_single_trial": ("fmrimod.single", "lsa_single_trial"),
    "lss_single_trial": ("fmrimod.single", "lss_single_trial"),
    "meta_effective_n": ("fmrimod.stats", "meta_effective_n"),
    "n_subjects": ("fmrimod.accessors", "n_subjects"),
    "p_values": ("fmrimod.accessors", "p_values"),
    "paired_diff_block": ("fmrimod.glm", "paired_diff_block"),
    "pvalues": ("fmrimod.accessors", "pvalues"),
    "r_to_z": ("fmrimod.stats", "r_to_z"),
    "replay": ("fmrimod.glm", "replay"),
    "replay_fits": ("fmrimod.glm", "replay_fits"),
    "ReplayContractError": ("fmrimod.glm", "ReplayContractError"),
    "ReplayResult": ("fmrimod.glm", "ReplayResult"),
    "ResultsManifest": ("fmrimod.io", "ResultsManifest"),
    "read_fmri_config": ("fmrimod.dataset", "read_fmri_config"),
    "read_h5_full": ("fmrimod.dataset", "read_h5_full"),
    "read_nifti_full": ("fmrimod.dataset", "read_nifti_full"),
    "register_basis": ("fmrimod.dataset", "register_basis"),
    "resolve_basis": ("fmrimod.dataset", "resolve_basis"),
    "samples": ("fmrimod.utils", "samples"),
    "sandwich_from_whitened_resid": ("fmrimod.ar", "sandwich_from_whitened_resid"),
    "se": ("fmrimod.accessors", "se"),
    "shift": ("fmrimod.utils", "shift"),
    "simulate_bold_signal": ("fmrimod.simulate", "simulate_bold_signal"),
    "simulate_fmri_matrix": ("fmrimod.simulate", "simulate_fmri_matrix"),
    "simulate_noise_vector": ("fmrimod.simulate", "simulate_noise_vector"),
    "simulate_simple_dataset": ("fmrimod.simulate", "simulate_simple_dataset"),
    "soft_projection": ("fmrimod.glm", "soft_projection"),
    "soft_subspace_options": ("fmrimod.model", "soft_subspace_options"),
    "standard_error": ("fmrimod.accessors", "standard_error"),
    "t_to_beta_se": ("fmrimod.glm", "t_to_beta_se"),
    "t_to_d": ("fmrimod.stats", "t_to_d"),
    "tidy": ("fmrimod.accessors", "tidy"),
    "tidy_fitted_hrf": ("fmrimod.accessors", "tidy_fitted_hrf"),
    "volume_weights": ("fmrimod.glm.preprocess", "volume_weights"),
    "voxel_index_chunks": ("fmrimod.dataset", "voxel_index_chunks"),
    "whiten": ("fmrimod.ar", "whiten"),
    "whiten_apply": ("fmrimod.ar", "whiten_apply"),
    "write_results": ("fmrimod.io", "write_results"),
    "z_to_r": ("fmrimod.stats", "z_to_r"),
    "zscores": ("fmrimod.accessors", "zscores"),
}


# NOTE: intentionally left unannotated. A PEP 562 module __getattr__ is
# what mypy uses to type every unresolved `fmrimod.X` / `from fmrimod
# import X` access. Annotating it `-> Any` propagates Any? into importers
# under the strict config and reintroduces "Any? not callable" errors in
# otherwise-clean modules (verified: glm/compat.py, dataset/compat.py).
# `-> object` is worse (breaks callable re-exports). The lone
# no-untyped-def here is cheaper than that cascade. (bd-01KRNQDEH9TXKKH9JVDVR71RFD)
def __getattr__(name: str):  # PEP 562  # noqa: ANN201
    """Resolve lazy top-level re-exports on first access."""
    entry = _LAZY_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module 'fmrimod' has no attribute {name!r}")
    module_path, attr = entry
    from importlib import import_module

    value = getattr(import_module(module_path), attr)
    globals()[name] = value
    return value


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
    # R-parity aliases for fmrihrf constant names
    "HRF_SPMG1",
    "HRF_SPMG2",
    "HRF_SPMG3",
    "HRF_GAMMA",
    "HRF_GAUSSIAN",
    "HRF_BSPLINE",
    "HRF_FIR",
    "HRF_FOURIER",
    "HRF_TIME",
    "HRF_MEXHAT",
    "HRF_INV_LOGIT",
    "HRF_HALF_COSINE",
    "HRF_SINE",
    "HRF_LWU",
    "HRF_LWU_BASIS",
    "spm_canonical",
    "gamma_hrf",
    "gaussian_hrf",
    # Spec / Term builders (``hrf`` and ``trialwise`` live under
    # ``fmrimod.spec`` because their names collide with existing submodules)
    "drift",
    "intercept",
    "confounds",
    "Spec",
    "Term",
    "HrfTerm",
    "Drift",
    "Intercept",
    "Confounds",
    "as_spec",
    "is_spec",
    "spec_diff",
    "SpecDiff",
    "TermDiff",
    "FieldDiff",
    "SpecSerializationError",
    # Contrast taxonomy (R-parity constructors)
    "pair_contrast",
    "unit_contrast",
    "oneway_contrast",
    "interaction_contrast",
    "poly_contrast",
    "column_contrast",
    "contrast_set",
    "pairwise_contrasts",
    "one_against_all_contrast",
    "sliding_window_contrasts",
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
    "check_nuisance",
    "clean_nuisance",
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
    "voxel_index_chunks",
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
    "ContrastDelta",
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
    "combine_runs",
    "combine_contrasts",
    "soft_subspace_options",
    "soft_projection",
    "apply_soft_projection",
    "compute_lm_contrasts",
    "compute_lm_contrasts_from_suffstats",
    "fit_contrasts",
    "fit_glm_on_transformed_series",
    "fit_glm_with_config",
    "fit_glm_from_matrix",
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
    "estimate_single_trial",
    "estimate_single_trial_from_dataset",
    "lss_single_trial",
    "lsa_single_trial",
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
    "ResultsManifest",
    "write_results",
    "t_to_d",
    "r_to_z",
    "replay",
    "replay_fits",
    "ReplayContractError",
    "ReplayResult",
    "z_to_r",
    "simulate_simple_dataset",
    "simulate_bold_signal",
    "simulate_noise_vector",
    "simulate_fmri_matrix",
    "estimate_hrf",
]
