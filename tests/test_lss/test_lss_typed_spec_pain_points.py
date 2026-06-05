"""Pain points pinned by the LSS single-trial parity workflow.

Two ergonomic gaps surfaced and one bug was fixed while wiring
``tier_a_single_trial_lss``. The bug fix is pinned positively; the
remaining gaps are pinned as current-state so a future fix can flip
the assertion to the desired state.
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


# -- Pain point: no typed lss_single_trial wrapper on fmri_lm ---------------


def test_lss_requires_manual_design_extraction_from_typed_spec() -> None:
    """Document the current LSS workflow: pull X and Z out by hand.

    There is no ``fmri_lm(..., engine="lss")`` or ``single_trial(
    method="lss")`` typed entry point. The user gets the per-trial
    X and the baseline Z out of a typed ``trialwise()`` fit and then
    calls the matrix-first ``lss_single_trial(Y, X, baseline_-
    regressors=Z)`` directly. Pinning the manual-extraction shape
    here so a future typed wrapper can substitute in cleanly.
    """
    from fmrimod.single import lss_single_trial

    fit, events = _trialwise_fit()
    full_design = fit.model.design_matrix_array(run=None)
    cols = fit.design_columns()

    trial_indices = [c.index for c in cols.where(term="trial").columns]
    baseline_indices = [
        c.index for c in cols.columns
        if c.role in ("drift", "intercept", "confound", "baseline")
    ]
    X = full_design[:, trial_indices]
    Z = full_design[:, baseline_indices]
    # Manufactured Y for the contract test.
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(full_design.shape[0], 4))

    result = lss_single_trial(
        Y=Y, X=X, baseline_regressors=Z, include_intercept=False,
    )
    assert result.betas.shape == (len(events), 4)


# -- Pain point: trial labels not surfaced on column provenance --------------


def test_trialwise_column_condition_is_trial_index_not_experimental_condition() -> None:
    """``trialwise()`` columns carry trial-index provenance, not condition labels.

    Each per-trial column has ``term='trial'``, ``level='1'``, ...
    ``level='N'``, and ``condition='trial.1'`` ... ``'trial.N'`` —
    so the trial INDEX is surfaced but the experimental condition
    label (``trial_type``: ``"A"`` or ``"B"`` here) is NOT. MVPA
    pipelines need the condition label to assemble per-trial betas
    into condition-specific decoding folds, and currently have to
    parse the trial index out of the column name and join back to
    the events DataFrame manually.

    A future fix could either:

    1. Add a new ``condition_label`` / ``stimulus`` provenance field
       carrying the original ``trial_type`` value.
    2. Or repurpose ``condition`` to mean the experimental condition
       and use a separate field (``trial_index``) for the index.

    Pinned at the current state so the future fix has a clear
    regression target.
    """
    fit, events = _trialwise_fit()
    cols = fit.design_columns()
    trial_cols = list(cols.where(term="trial").columns)

    # Current state: condition is "trial.{k}" — the trial index, not
    # the experimental condition.
    for c in trial_cols:
        assert (c.condition or "").startswith("trial."), (
            f"current state: condition encodes the trial index as "
            f"'trial.{{k}}'; got {c.condition!r}. If a future fix changes "
            f"this to carry the experimental condition label, update the "
            f"regression to assert the correct value (e.g. condition in "
            f"('A', 'B'))."
        )

    # Current workaround: parse trial index out of the name and join
    # back via the events DataFrame. This is what an MVPA pipeline
    # has to do today.
    sorted_events = events.sort_values("onset").reset_index(drop=True)
    parsed_indices = sorted(
        int(c.name.removeprefix("trial_")) for c in trial_cols
    )
    assert parsed_indices == list(range(1, len(events) + 1))
    inferred_conditions = [
        sorted_events.iloc[i - 1]["trial_type"] for i in parsed_indices
    ]
    assert all(cond in ("A", "B") for cond in inferred_conditions)
