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
from typing import TYPE_CHECKING, Any, Optional, cast

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
    SpatialDescriptor,
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
    # Optional prewhitening. Routes through fmrimod.ar.fit_noise +
    # whiten_apply (the same engine the GLM solver uses) so Y, X,
    # baseline_regressors, and confounds share one plan.
    whitening_plan = None
    if prewhiten is not None and prewhiten.method != "none":
        from ._prewhiten import prewhiten_matrices
        pw = prewhiten_matrices(
            Y, X, confounds, prewhiten,
            baseline_regressors=baseline_regressors,
        )
        Y, X, confounds = pw.Y, pw.X, pw.confounds
        baseline_regressors = pw.baseline_regressors
        whitening_plan = pw.plan

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
        result = lss_single_trial(
            Y, X,
            confounds=confounds,
            nuisance_projector=nuisance_projector,
            chunk_size=chunk_size,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )
    elif method_enum is SingleTrialMethod.LSA:
        result = lsa_single_trial(
            Y, X,
            confounds=confounds,
            return_se=return_se,
            trial_labels=trial_labels,
            baseline_regressors=baseline_regressors,
            include_intercept=include_intercept,
        )
    elif method_enum is SingleTrialMethod.OASIS:
        oasis_cfg = oasis_config or OasisConfig(return_se=return_se)
        if return_se and not oasis_cfg.return_se:
            oasis_cfg = replace(oasis_cfg, return_se=True)
        result = oasis_single_trial(
            Y, X,
            confounds=confounds,
            config=oasis_cfg,
            trial_labels=trial_labels,
        )
    elif method_enum is SingleTrialMethod.SBHM:
        # Lazy import to avoid circular dependencies
        from .sbhm.pipeline import sbhm_single_trial
        sbhm_cfg = sbhm_config or SbhmConfig()
        result = sbhm_single_trial(
            Y, X,
            confounds=confounds,
            config=sbhm_cfg,
            trial_labels=trial_labels,
            library=sbhm_library,
        )
    elif method_enum is SingleTrialMethod.MIXED:
        from .mixed import mixed_single_trial
        result = mixed_single_trial(
            Y, X,
            confounds=confounds,
            trial_labels=trial_labels,
        )
    elif method_enum is SingleTrialMethod.LSS_VOXEL_HRF:
        raise ValueError(
            "method='lss_voxel_hrf' requires an estimated VoxelHrfResult; "
            "call lss_with_voxel_hrf(...) directly."
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if whitening_plan is not None:
        result.extra["whitening_plan"] = whitening_plan
    return result


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
        names plus dataset-derived metadata: ``trial_table`` (the events
        slice aligned with ``betas`` rows), ``run_labels`` (per-trial
        block/run id when the events table carries one), ``subject_id``
        (when the dataset advertises one), and ``spatial_descriptor``
        (the capability label describing the voxel space the betas live
        in). Matrix-first :func:`estimate_single_trial` callers continue
        to leave all four metadata fields ``None``.
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

    sampling_frame = cast(Any, dataset).get_sampling_frame()

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

    Y = np.asarray(cast(Any, dataset).get_data(), dtype=np.float64)

    result = estimate_single_trial(
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

    _attach_dataset_metadata(
        result,
        dataset=dataset,
        events_df=events_df,
        block_column=resolved_block if isinstance(resolved_block, str) else None,
    )
    return result


def _resolve_subject_id(dataset: object) -> object | None:
    """Look up the dataset's subject identifier, if it carries one.

    Checks ``dataset.subject_id`` first (single-subject datasets), then
    falls back to the singleton entry of ``dataset.subject_ids`` (typed
    multi-subject containers that nonetheless wrap one subject). Returns
    ``None`` if neither surface is present or the latter is empty/has
    multiple entries (the single-trial wrapper currently fits one subject
    at a time, so an ambiguous multi-entry list is treated as "no id").
    """
    direct = getattr(dataset, "subject_id", None)
    if direct is not None:
        return cast(object, direct)
    ids = getattr(dataset, "subject_ids", None)
    if ids is None:
        return None
    try:
        seq = list(ids)
    except TypeError:
        return None
    if len(seq) == 1:
        return cast(object, seq[0])
    return None


def _build_spatial_descriptor(
    dataset: object, n_voxels: int
) -> "SpatialDescriptor | None":
    """Construct a :class:`SpatialDescriptor` from the dataset's mask, if any."""
    get_mask = getattr(dataset, "get_mask", None)
    if not callable(get_mask):
        return None
    try:
        mask = np.asarray(get_mask())
    except Exception:
        return None
    return SpatialDescriptor(
        n_voxels=int(n_voxels),
        mask_shape=tuple(int(d) for d in mask.shape),
        mask_n_true=int(np.asarray(mask, dtype=bool).sum()),
    )


def _attach_dataset_metadata(
    result: "SingleTrialResult",
    *,
    dataset: object,
    events_df: object,
    block_column: str | None,
) -> None:
    """Populate dataset-derived metadata fields on ``result`` in-place.

    Mutates the ``SingleTrialResult`` returned by the matrix-first
    dispatcher so the dataset-path wrapper carries trial/run/subject
    metadata plus a spatial-identity label. No-op for fields the
    dataset cannot supply (e.g. missing block column, no mask).
    """
    n_trials = int(result.betas.shape[0])
    n_voxels = int(result.betas.shape[1]) if result.betas.ndim == 2 else 0

    # Trial table is meaningful only when events rows map 1:1 to trials.
    # Multi-basis trialwise (n_trials * K rows of betas) leaves the table
    # field None rather than risk a misaligned slice.
    events_any = cast(Any, events_df)
    try:
        n_rows = int(len(events_any))
    except TypeError:
        n_rows = -1
    if n_rows == n_trials:
        try:
            result.trial_table = events_any.reset_index(drop=True).copy()
        except Exception:
            result.trial_table = None
        if block_column is not None and block_column in events_any.columns:
            result.run_labels = tuple(events_any[block_column].tolist())

    result.subject_id = _resolve_subject_id(dataset)
    result.spatial_descriptor = _build_spatial_descriptor(dataset, n_voxels)


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
    "SpatialDescriptor",
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
