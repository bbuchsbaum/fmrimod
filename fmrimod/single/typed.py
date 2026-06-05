"""Typed-Spec entry point for single-trial estimation.

This module exposes :func:`fmri_lss`, a typed counterpart to
:func:`fmrimod.fmri_lm` that compiles a typed
:class:`~fmrimod.spec.Spec` containing a :func:`~fmrimod.spec.trialwise`
term, extracts the per-trial design and baseline regressors from the
realised model, and runs Mumford LSS via the vectorised
:func:`lss_single_trial` solver.

The legacy formula-API counterpart is
:func:`fmrimod.single.estimate_single_trial_from_dataset` which still
takes a string formula or a list of
:class:`~fmrimod.formula.base.Term` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._types import SingleTrialResult
from .lss import lss_single_trial

if TYPE_CHECKING:
    from ..dataset.fmri_dataset import FmriDataset
    from ..spec import Spec, Term


def fmri_lss(
    spec: "Spec | Term | Sequence[object] | str",
    dataset: "FmriDataset",
    *,
    block: Optional[Union[str, NDArray[np.float64]]] = None,
    durations: Optional[Union[str, float, NDArray[np.float64]]] = None,
    precision: Optional[float] = None,
    return_se: bool = False,
    chunk_size: Optional[int] = None,
) -> SingleTrialResult:
    """Run Mumford LSS single-trial estimation through the typed Spec surface.

    The headline shape::

        spec = (
            trialwise(basis="spm")
            + drift("poly", degree=2)
            + intercept(per="run")
        )
        result = fm.fmri_lss(spec, dataset)
        result.betas   # (n_trials, n_voxels) per-trial single-trial estimates

    The compile step realises the design, the per-trial columns are
    pulled out via the typed ``cols.where(term="trial")`` lookup, the
    drift / intercept / confound columns become the baseline-projection
    regressors, and the data matrix is read from ``dataset``.
    :func:`lss_single_trial` then runs the inner LSS loop in vectorised
    form.

    Parameters
    ----------
    spec
        A typed :class:`~fmrimod.spec.Spec` containing exactly one
        :func:`~fmrimod.spec.trialwise` term plus optional baseline
        terms (``drift``, ``intercept``, ``confounds``). Legacy formula
        strings and ``list[Term]`` inputs are also accepted — they go
        through the same lowering as :func:`fmrimod.fmri_lm`.
    dataset
        :class:`~fmrimod.dataset.FmriDataset` carrying the data,
        events, and sampling frame.
    block, durations, precision
        Forwarded to the typed compile path. See :func:`fmrimod.fmri_lm`
        for semantics.
    return_se
        If ``True``, compute per-trial standard errors.
    chunk_size
        Voxel-chunk size for the vectorised LSS solve. ``None`` lets
        :func:`lss_single_trial` pick a heuristic.

    Returns
    -------
    SingleTrialResult
        ``betas`` has shape ``(n_trials, n_voxels)``. ``trial_labels``
        carries the realised column names from the trialwise term
        (``"trial_01"``, ``"trial_02"``, ...). ``residual_df`` carries
        the per-trial degrees of freedom (``n - 2 - rank(baseline)``).
    """
    # Lazy imports so this module stays cheap to import.
    from ..glm.fmri_lm import _build_model_from_spec

    model = _build_model_from_spec(
        spec=spec,
        dataset=dataset,
        baseline=None,
        block=block,
        durations=durations,
        precision=precision,
    )

    # Realise the design and partition into trial X and baseline Z.
    X_full = np.asarray(
        model.design_matrix_array(run=None), dtype=np.float64
    )
    from ..design.columns import DesignColumns

    cols = DesignColumns.from_model(model)
    trial_indices = [c.index for c in cols.where(term="trial").columns]
    if not trial_indices:
        raise ValueError(
            "fmri_lss(spec, dataset): the typed spec must contain exactly "
            "one trialwise() term producing per-trial regressors. None were "
            "found in the realised design."
        )
    baseline_indices = [
        c.index for c in cols.columns
        if c.role in ("drift", "intercept", "confound", "baseline")
    ]

    X = X_full[:, trial_indices]
    Z = (
        X_full[:, baseline_indices]
        if baseline_indices
        else None
    )

    # Per-trial labels surfaced as ``trial_01`` etc.; use the realised
    # column names so the result carries them.
    trial_labels = [
        cols.columns[i].name for i in trial_indices
    ]

    # Gather concatenated data from the dataset.
    Y = _gather_dataset_data(dataset)
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            f"fmri_lss: dataset row count ({Y.shape[0]}) does not match "
            f"realised design row count ({X.shape[0]})"
        )

    return lss_single_trial(
        Y=Y,
        X=X,
        baseline_regressors=Z,
        include_intercept=False,
        return_se=return_se,
        chunk_size=chunk_size,
        trial_labels=trial_labels,
    )


def _gather_dataset_data(dataset: "FmriDataset") -> NDArray[np.float64]:
    """Return the run-concatenated ``timepoints x voxels`` data matrix."""
    matrix_fn = getattr(dataset, "get_data_matrix", None)
    if callable(matrix_fn):
        return np.asarray(matrix_fn(), dtype=np.float64)
    n_runs = int(getattr(dataset, "n_runs", 1))
    parts = []
    for r in range(n_runs):
        parts.append(np.asarray(dataset.get_data(r), dtype=np.float64))
    return np.vstack(parts)
