"""Single-trial beta estimation for fMRI.

Provides multiple algorithms for estimating per-trial activation
amplitudes from event-related fMRI designs:

- **LSS** (Least Squares Separate): vectorized closed-form solver
- **LSA** (Least Squares All): standard OLS with all trials
- **OASIS**: optimised batched LSS with ridge regularisation
- **SBHM**: Shared-Basis HRF Matching pipeline
- **Mixed**: mixed-model solver

The primary entry point is :func:`estimate_single_trial`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from ._prewhiten import PrewhitenConfig
from ._project import NuisanceProjector, build_nuisance_projector
from ._types import (
    OasisConfig,
    SbhmConfig,
    SingleTrialMethod,
    SingleTrialMethodLike,
    SingleTrialResult,
    VoxelHrfResult,
)
from .item import (
    ItemBundle,
    ItemCovarianceBlockResult,
    ItemCovarianceMatrixResult,
    ItemCovarianceResult,
    ItemCvAggregate,
    ItemCvResult,
    ItemFoldMetrics,
    ItemFoldSplit,
    ItemPredictions,
    ItemWeightsResult,
    item_build_design,
    item_compute_u,
    item_cv,
    item_fit,
    item_from_lsa,
    item_predict,
    item_slice_fold,
)
from .lsa import lsa_single_trial
from .lss import lss_single_trial
from .mixed import mixed_single_trial
from .oasis import oasis_single_trial
from .voxel_hrf import estimate_hrf, estimate_voxel_hrf, lss_with_voxel_hrf

if TYPE_CHECKING:
    from .sbhm.library import SbhmLibrary


def estimate_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    method: SingleTrialMethodLike = "lss",
    confounds: NDArray[np.float64] | None = None,
    trial_labels: list[str] | None = None,
    return_se: bool = False,
    *,
    prewhiten: PrewhitenConfig | None = None,
    nuisance_projector: NuisanceProjector | None = None,
    chunk_size: int | None = None,
    oasis_config: OasisConfig | None = None,
    sbhm_config: SbhmConfig | None = None,
    sbhm_library: SbhmLibrary | None = None,
    baseline_regressors: NDArray[np.float64] | None = None,
    include_intercept: bool = False,
) -> SingleTrialResult:
    """Estimate per-trial betas using the specified method.

    This is the main dispatcher for single-trial estimation.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix (time x voxels).
    X : NDArray, shape ``(n, n_trials)`` or ``(n, n_trials * K)``
        Trial regressor matrix (already convolved with HRF).
    method : SingleTrialMethod or literal string
        One of ``"lss"``, ``"lsa"``, ``"oasis"``, ``"sbhm"``,
        ``"mixed"``.
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors (motion, drift, etc.).
    trial_labels : list of str, optional
        Labels identifying each trial.
    return_se : bool
        Whether to compute standard errors (not all methods support
        this).
    prewhiten : PrewhitenConfig, optional
        If provided, AR-based prewhitening is applied to Y, X, and
        confounds before estimation.
    nuisance_projector : NuisanceProjector, optional
        Precomputed nuisance projector for LSS. Useful when reusing the
        same confound matrix across repeated LSS fits.
    chunk_size : int, optional
        Voxel chunk size for LSS beta-only solves.
    oasis_config : OasisConfig, optional
        Configuration for the OASIS solver.
    sbhm_config : SbhmConfig, optional
        Configuration for the SBHM pipeline.
    sbhm_library : SbhmLibrary, optional
        Pre-built SBHM library (required for SBHM method).
    baseline_regressors : NDArray, shape ``(n, p)``, optional
        Baseline or experimental regressors included in every LSS model.
        Equivalent to the ``Z`` matrix in ``fmrilss::lss``.
    include_intercept : bool
        Add an intercept to the LSS adjustment design.

    Returns
    -------
    SingleTrialResult
    """
    # Optional prewhitening
    if prewhiten is not None and prewhiten.method != "none":
        from ._prewhiten import prewhiten_matrices
        Y, X, confounds = prewhiten_matrices(Y, X, confounds, prewhiten)

    try:
        method_enum = (
            method
            if isinstance(method, SingleTrialMethod)
            else SingleTrialMethod(method)
        )
    except ValueError as exc:
        valid = ", ".join(member.value for member in SingleTrialMethod)
        raise ValueError(f"method must be one of: {valid}") from exc
    if method_enum not in {SingleTrialMethod.LSS, SingleTrialMethod.LSA} and (
        baseline_regressors is not None or include_intercept
    ):
        raise ValueError(
            "baseline_regressors and include_intercept are currently supported "
            "only for method='lss' or method='lsa'."
        )

    if method_enum is SingleTrialMethod.LSS:
        return lss_single_trial(
            Y, X,
            confounds=confounds,
            nuisance_projector=nuisance_projector,
            chunk_size=chunk_size,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )

    if method_enum is SingleTrialMethod.LSA:
        return lsa_single_trial(
            Y, X,
            confounds=confounds,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )

    if method_enum is SingleTrialMethod.OASIS:
        oasis_cfg = oasis_config or OasisConfig(return_se=return_se)
        if return_se and not oasis_cfg.return_se:
            oasis_cfg = replace(oasis_cfg, return_se=True)
        return oasis_single_trial(
            Y, X,
            confounds=confounds,
            config=oasis_cfg,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.SBHM:
        # Lazy import to avoid circular dependencies
        from .sbhm.pipeline import sbhm_single_trial
        sbhm_cfg = sbhm_config or SbhmConfig()
        return sbhm_single_trial(
            Y, X,
            confounds=confounds,
            config=sbhm_cfg,
            trial_labels=trial_labels,
            library=sbhm_library,
        )

    if method_enum is SingleTrialMethod.MIXED:
        from .mixed import mixed_single_trial
        return mixed_single_trial(
            Y, X,
            confounds=confounds,
            trial_labels=trial_labels,
        )

    if method_enum is SingleTrialMethod.LSS_VOXEL_HRF:
        raise ValueError(
            "method='lss_voxel_hrf' requires an estimated VoxelHrfResult; "
            "call lss_with_voxel_hrf(...) directly."
        )

    raise ValueError(f"Unknown method: {method!r}")


def estimate_single_trial_from_dataset(
    dataset: object,
    spec: object,
    *,
    method: SingleTrialMethodLike = "lss",
    confounds: NDArray[np.float64] | None = None,
    return_se: bool = False,
    block: object = None,
    durations: object = None,
    precision: float | None = None,
    prewhiten: PrewhitenConfig | None = None,
    nuisance_projector: NuisanceProjector | None = None,
    chunk_size: int | None = None,
    oasis_config: OasisConfig | None = None,
    sbhm_config: SbhmConfig | None = None,
    sbhm_library: SbhmLibrary | None = None,
    include_intercept: bool = False,
) -> SingleTrialResult:
    """Estimate per-trial betas from an FmriDataset and trialwise spec.

    Public-seam alternative to :func:`estimate_single_trial` that consumes a
    :class:`~fmrimod.dataset.FmriDataset` plus a spec containing a
    :func:`~fmrimod.trialwise` construction, builds the trial design matrix
    from spec + dataset events, and delegates to
    :func:`estimate_single_trial`.

    Parameters
    ----------
    dataset : FmriDataset
        Typed dataset carrying time-series data and an event table.
    spec : str or list of Term or EventModelBuilder
        Trialwise specification. Slice 1 accepts whatever
        :func:`fmrimod.event_model` accepts: a string formula such as
        ``"trialwise()"`` or ``"trialwise(add_sum=True)"``, a list of
        :class:`~fmrimod.formula.base.Term` objects, or a builder.
        Routing typed :class:`~fmrimod.spec.Spec`/``Term`` trees through this
        wrapper waits on the typed compile path learning the trialwise
        placeholder lowering (separate follow-on slice).
    method : SingleTrialMethod or literal string
        Estimator selector; see :func:`estimate_single_trial`.
    confounds, return_se, prewhiten, nuisance_projector, chunk_size,
        oasis_config, sbhm_config, sbhm_library, include_intercept
        Forwarded to :func:`estimate_single_trial`.
    block, durations, precision
        Forwarded to :func:`fmrimod.event_model`. ``block`` and ``durations``
        default to event-table column auto-detection (``run``/``block`` and
        ``duration``).

    Returns
    -------
    SingleTrialResult
        Carries ``trial_labels`` derived from the trialwise term column
        names. Richer trial/run/subject metadata and spatial-identity
        capability label are deferred to follow-on slices of
        ``bd-01KRGQCT34QWSYKQ38BVFHD51E``.
    """
    # Build the EventModel via the design module directly. The typed
    # fmrimod.spec compile path does not yet special-case trialwise() terms
    # (the _trialwise_placeholder_ event is only resolved by
    # fmrimod.design.event_model); routing trialwise designs through the
    # established design entry point is the working path for Slice 1.
    # Imports are local to avoid a fmrimod.single → fmrimod.design import
    # cycle at module load.
    from ..design.event_model import event_model as _build_event_model

    events_df = getattr(dataset, "event_table", None)
    if events_df is None:
        raise ValueError(
            "estimate_single_trial_from_dataset(dataset, spec) requires the "
            "dataset to carry an event table; pass events= to "
            "fmri_dataset(...) so the trialwise spec can be compiled."
        )

    sampling_frame = dataset.get_sampling_frame()

    resolved_block = block
    if resolved_block is None:
        if "run" in events_df.columns:
            resolved_block = "run"
        elif "block" in events_df.columns:
            resolved_block = "block"
        else:
            resolved_block = np.ones(len(events_df), dtype=int)

    resolved_durations = durations
    if resolved_durations is None and "duration" in events_df.columns:
        resolved_durations = "duration"

    event_model = _build_event_model(
        formula=spec,
        data=events_df,
        block=resolved_block,
        sampling_frame=sampling_frame,
        durations=resolved_durations,
        precision=precision,
    )

    trial_labels = _extract_trialwise_labels(event_model)
    if not trial_labels:
        raise ValueError(
            "estimate_single_trial_from_dataset(dataset, spec) requires the "
            "spec to contain a trialwise() term; the compiled event model "
            "has no trialwise columns."
        )

    X = np.ascontiguousarray(
        np.asarray(event_model.design_matrix, dtype=np.float64)
    )
    baseline_regressors: NDArray[np.float64] | None = None

    Y = np.asarray(dataset.get_data(), dtype=np.float64)

    return estimate_single_trial(
        Y, X,
        method=method,
        confounds=confounds,
        trial_labels=trial_labels,
        return_se=return_se,
        prewhiten=prewhiten,
        nuisance_projector=nuisance_projector,
        chunk_size=chunk_size,
        oasis_config=oasis_config,
        sbhm_config=sbhm_config,
        sbhm_library=sbhm_library,
        baseline_regressors=baseline_regressors,
        include_intercept=include_intercept,
    )


def _extract_trialwise_labels(event_model: object) -> list[str] | None:
    """Return the column names belonging to trialwise term(s) in ``event_model``.

    Returns ``None`` when no trialwise terms are present; an empty list is
    treated as the same signal.
    """
    terms = getattr(event_model, "terms", None)
    if not terms:
        return None
    column_indices = getattr(event_model, "column_indices", None)
    if column_indices is None:
        return None
    column_names = getattr(event_model, "column_names", None)
    if column_names is None:
        return None
    labels: list[str] = []
    for term in terms:
        if not getattr(term, "_is_trialwise", False):
            continue
        positions = column_indices.get(getattr(term, "name", None))
        if positions is None:
            continue
        for i in positions:
            labels.append(column_names[i])
    return labels or None


__all__ = [
    "estimate_single_trial",
    "estimate_single_trial_from_dataset",
    "lss_single_trial",
    "lsa_single_trial",
    "oasis_single_trial",
    "mixed_single_trial",
    "item_build_design",
    "item_compute_u",
    "item_fit",
    "item_predict",
    "item_slice_fold",
    "item_cv",
    "item_from_lsa",
    "estimate_hrf",
    "estimate_voxel_hrf",
    "lss_with_voxel_hrf",
    "SingleTrialResult",
    "SingleTrialMethod",
    "SingleTrialMethodLike",
    "OasisConfig",
    "SbhmConfig",
    "PrewhitenConfig",
    "VoxelHrfResult",
    "NuisanceProjector",
    "build_nuisance_projector",
    "ItemBundle",
    "ItemCovarianceResult",
    "ItemCovarianceMatrixResult",
    "ItemCovarianceBlockResult",
    "ItemWeightsResult",
    "ItemFoldSplit",
    "ItemFoldMetrics",
    "ItemPredictions",
    "ItemCvAggregate",
    "ItemCvResult",
]
