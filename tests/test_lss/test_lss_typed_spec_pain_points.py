"""Regression tests pinning the typed LSS single-trial contract.

History: the ``tier_a_single_trial_lss`` parity workflow surfaced
three issues which were all fixed in the same series of commits.
This suite pins the fixed state so the typed LSS surface can't
silently regress.

1. **Typed ``trialwise()`` lowering** — the sentinel
   ``variables=("__trial__",)`` is now translated to
   ``_is_trialwise=True`` on the EventModelTerm.
2. **Typed ``fmri_lss(spec, dataset)`` wrapper** — one-call LSS
   that compiles the spec, extracts trial X + baseline Z, and
   runs the vectorised LSS solver.
3. **Per-trial experimental-condition labels** —
   ``trialwise(condition="trial_type")`` populates
   ``DesignColumn.condition`` with each trial's experimental
   condition value, so MVPA pipelines can group per-trial betas
   via ``cols.where(role="task", condition="A")``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.spec import drift, intercept, trialwise


def _trialwise_fit():
    """Build a typed trialwise fit on a small interleaved A/B design."""
    rng = np.random.default_rng(0)
    n_trials = 8
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 90.0, n_trials),
        "duration": 0.0,
        "trial_type": ["A", "B"] * (n_trials // 2),
        "run": 1,
    })
    ds = fm.fmri_dataset(
        np.zeros((80, 4)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    spec = (
        trialwise(basis="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fm.fmri_lm(spec, ds), events


# -- Fixed: typed trialwise() spec resolves correctly -----------------------


def test_trialwise_typed_spec_resolves_per_trial_columns() -> None:
    """The typed ``trialwise()`` builder must lower to a working design.

    Before the fix, ``fmri_lm(trialwise(...), ds)`` raised
    ``ValueError: Event '__trial__' not found in model`` because
    ``_hrf_term_to_event_model_term`` didn't propagate the
    ``_is_trialwise`` marker from the sentinel
    ``variables=("__trial__",)``. Now the lowering detects the sentinel
    and tags the EventModelTerm correctly.
    """
    fit, events = _trialwise_fit()
    cols = fit.design_columns()
    trial_cols = list(cols.where(term="trial").columns)
    assert len(trial_cols) == len(events), (
        f"expected {len(events)} trial columns from trialwise(); got "
        f"{len(trial_cols)}"
    )
    for c in trial_cols:
        assert c.role == "task", (
            f"trialwise columns should carry role='task'; got {c.role!r}"
        )
        assert c.name and c.name.startswith("trial_"), (
            f"unexpected trialwise column name {c.name!r}"
        )


# -- Fixed: typed fmri_lss(spec, dataset) one-call wrapper -----------------


def test_fmri_lss_typed_wrapper_one_call_workflow() -> None:
    """``fm.fmri_lss(spec, ds)`` runs LSS end-to-end from the typed spec.

    Before the fix the user had to:
    1. Build a fit via ``fm.fmri_lm(spec, ds)``.
    2. Extract trial X via ``cols.where(term="trial")``.
    3. Extract baseline Z via the baseline roles.
    4. Pull Y out of the dataset by hand.
    5. Call the matrix-first ``lss_single_trial`` directly.

    Now ``fm.fmri_lss(spec, ds)`` does the whole pipeline and returns a
    :class:`SingleTrialResult` with ``betas`` shape
    ``(n_trials, n_voxels)`` plus trial labels and DoF.
    """
    rng = np.random.default_rng(2)
    n_trials = 8
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 90.0, n_trials),
        "duration": 0.0,
        "trial_type": ["A", "B"] * (n_trials // 2),
        "run": 1,
    })
    data = rng.normal(size=(80, 4))
    ds = fm.fmri_dataset(
        data, tr=2.0, events=events, slice_timing_offset=0.0
    )
    spec = (
        trialwise(basis="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = fm.fmri_lss(spec, ds)

    assert result.betas.shape == (n_trials, 4)
    assert result.trial_labels is not None
    assert len(result.trial_labels) == n_trials
    assert all(label.startswith("trial_") for label in result.trial_labels)


def test_fmri_lss_matches_manual_extraction_path() -> None:
    """The typed wrapper produces the same betas as the manual path.

    Equivalence pin so a future refactor of either path can't drift.
    """
    from fmrimod.single import lss_single_trial

    rng = np.random.default_rng(3)
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 90.0, 8),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 4,
        "run": 1,
    })
    data = rng.normal(size=(80, 4))
    ds = fm.fmri_dataset(
        data, tr=2.0, events=events, slice_timing_offset=0.0
    )
    spec = (
        trialwise(basis="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        typed_result = fm.fmri_lss(spec, ds)

        # Manual extraction path.
        fit = fm.fmri_lm(spec, ds)
        full_X = fit.model.design_matrix_array(run=None)
        cols = fit.design_columns()
        trial_idx = [c.index for c in cols.where(term="trial").columns]
        baseline_idx = [
            c.index for c in cols.columns
            if c.role in ("drift", "intercept", "confound", "baseline")
        ]
        manual_result = lss_single_trial(
            Y=data,
            X=full_X[:, trial_idx],
            baseline_regressors=full_X[:, baseline_idx],
            include_intercept=False,
        )
    np.testing.assert_allclose(
        typed_result.betas, manual_result.betas, atol=1e-12
    )


# -- Pain point: trial labels not surfaced on column provenance --------------


# -- Fixed: per-trial experimental-condition labels -------------------------


def test_trialwise_default_condition_is_trial_index() -> None:
    """Without ``condition=``, ``trialwise()`` columns carry trial-index labels.

    Pinned for backward-compat: when no condition column is supplied,
    the realised per-trial columns get ``condition='trial.{k}'`` and
    ``level='{k}'`` (the trial index). MVPA users who don't supply
    ``condition=`` retain the legacy behavior they'd see in older
    versions of fmrimod.
    """
    fit, events = _trialwise_fit()
    cols = fit.design_columns()
    trial_cols = list(cols.where(term="trial").columns)
    for c in trial_cols:
        assert (c.condition or "").startswith("trial."), (
            f"default condition tag should be 'trial.{{k}}'; got {c.condition!r}"
        )


def test_trialwise_with_condition_kwarg_surfaces_experimental_label() -> None:
    """``trialwise(condition="trial_type")`` populates DesignColumn.condition.

    Each per-trial column now carries the user's experimental condition
    label (here ``"A"`` or ``"B"``) as ``DesignColumn.condition``, so
    typed lookup ``cols.where(role="task", condition="A")`` returns the
    trial-A subset directly — no name-parsing required.
    """
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 90.0, 8),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 4,
        "run": 1,
    })
    ds = fm.fmri_dataset(
        np.zeros((80, 4)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            trialwise(basis="spm", condition="trial_type")
            + intercept(per="run"),
            ds,
        )
    cols = fit.design_columns()
    trial_cols = list(cols.where(term="trial").columns)
    assert len(trial_cols) == 8
    # The condition labels alternate A, B, A, B, ... matching the events row order.
    expected_conditions = events.sort_values("onset")["trial_type"].tolist()
    actual_conditions = [c.condition for c in trial_cols]
    assert actual_conditions == expected_conditions

    # Typed condition-based lookup works.
    a_cols = list(cols.where(role="task", condition="A").columns)
    b_cols = list(cols.where(role="task", condition="B").columns)
    assert len(a_cols) == 4
    assert len(b_cols) == 4
    assert all(c.condition == "A" for c in a_cols)
    assert all(c.condition == "B" for c in b_cols)
