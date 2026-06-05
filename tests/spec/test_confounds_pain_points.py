"""Regression tests pinning ergonomic gaps in the typed ``confounds(...)`` API.

Surfaced while wiring the ``tier_a_realistic_confounds`` parity
workflow. Each test pins the *current* observed behavior so a
future fix can flip the assertion from "current state" to "desired
state". These are NOT correctness gaps — confound values plumb
through to the realised design bitwise — they are *ergonomic* gaps
in the typed surface that make confound-aware analyses harder than
they should be.

1. **No distinct ``role="confound"``** — confound columns share
   ``role="baseline"`` with intercept and drift. Filtering them
   from the column registry requires name-suffix parsing.
2. **No ``where(name="trans_x")`` direct match** — names get
   prefixed (``"nuis_runK_<name>"``) so the user's original
   DataFrame column name is not addressable in the typed lookup.
3. **Multi-run confounds via the typed spec** —
   ``confounds(source=concat_df)`` raises on multi-run designs;
   no public typed-spec path takes per-run DataFrames. The
   underlying ``baseline_model(nuisance_list=[df1, df2])``
   supports it; the gap is at the typed Spec surface.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset.constructors import matrix_dataset
from fmrimod.spec import confounds, hrf, intercept


def _single_run_fit() -> object:
    """Build a single-run fit with motion regressors as confounds."""
    rng = np.random.default_rng(0)
    n = 80
    events = pd.DataFrame({
        "onset": np.linspace(8.0, 96.0, 8),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 4,
        "run": 1,
    })
    conf_df = pd.DataFrame({
        "trans_x": rng.normal(scale=0.5, size=n),
        "rot_x":   rng.normal(scale=0.01, size=n),
    })
    ds = fm.fmri_dataset(
        np.zeros((n, 4)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds("trans_x", "rot_x", source=conf_df)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fm.fmri_lm(spec, ds, engine="concat")


# -- Pain point 1: no distinct role="confound" ------------------------------


def test_confound_columns_share_baseline_role_with_drift_intercept() -> None:
    """Confound columns currently report ``role="baseline"``.

    Pinned at the current state so a future fix that introduces a
    distinct ``role="confound"`` (or a similar refinement) can flip
    this assertion to the desired state with a clear diff.
    """
    fit = _single_run_fit()
    cols = fit.design_columns()
    confound_cols = [
        c for c in cols.columns
        if (c.name or "").startswith("nuis_")
    ]
    assert confound_cols, "expected at least one confound column"
    for c in confound_cols:
        # Current state: confounds carry the same role as drift/intercept.
        # If this test starts failing, the typed surface has differentiated
        # confound from baseline-drift — update accordingly.
        assert c.role == "baseline", (
            f"confound column {c.name!r} role changed from 'baseline' to "
            f"{c.role!r}; if intentional, update this regression"
        )


# -- Pain point 2: original column names get prefixed -----------------------


def test_confound_user_name_is_not_directly_addressable() -> None:
    """``cols.where(name="trans_x")`` does NOT match the confound column.

    The realised column carries a prefixed name (``"nuis_run1_trans_x"``)
    so the user's original DataFrame column name is buried. Until a
    typed-name lookup lands, users must suffix-match.
    """
    fit = _single_run_fit()
    cols = fit.design_columns()
    # Direct match on the user-visible name returns nothing.
    direct = [c for c in cols.columns if c.name == "trans_x"]
    assert direct == [], (
        f"direct name='trans_x' match should be empty (the column "
        f"is prefixed to 'nuis_runN_trans_x'); got {direct}"
    )
    # The suffix-match workaround does find it.
    suffix = [c for c in cols.columns if (c.name or "").endswith("trans_x")]
    assert len(suffix) == 1, (
        f"suffix-match workaround should find exactly one column; got "
        f"{[(c.index, c.name) for c in suffix]}"
    )
    assert "trans_x" in (suffix[0].name or "")


# -- Pain point 3: multi-run confounds via the typed spec --------------------


def test_typed_confounds_source_df_fails_on_multirun() -> None:
    """``confounds(source=concat_df)`` raises on a multi-run design.

    Pinned to surface the gap. The error currently reaches the user as
    a baseline-model nuisance-length mismatch. A future fix that adds
    a per-run list path to the typed Spec should change this assertion
    to the desired behavior.
    """
    rng = np.random.default_rng(0)
    n1, n2 = 60, 60
    events = pd.DataFrame({
        "onset": [10.0, 30.0, 10.0, 30.0],
        "duration": 0.0,
        "trial_type": ["A", "B", "A", "B"],
        "run": [1, 1, 2, 2],
    })
    conf_df = pd.DataFrame({
        "trans_x": rng.normal(size=n1 + n2),
    })
    ds = matrix_dataset(
        np.zeros((n1 + n2, 1)),
        tr=2.0,
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds("trans_x", source=conf_df)
        + intercept(per="run")
    )
    with pytest.raises(ValueError, match="nuisance_list"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, engine="concat")


def test_baseline_model_supports_multirun_nuisance_list_directly() -> None:
    """The underlying engine does support per-run confounds.

    This is the dual of the previous test: it pins that the gap is
    purely in the typed-Spec surface, not in the lower
    ``baseline_model`` API. A future fix at the Spec level can build
    on the working baseline_model path without breaking anything.
    """
    rng = np.random.default_rng(0)
    n1, n2 = 40, 40
    sf = fm.SamplingFrame(blocklens=[n1, n2], TR=2.0)
    df1 = pd.DataFrame({"trans_x": rng.normal(size=n1)})
    df2 = pd.DataFrame({"trans_x": rng.normal(size=n2)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = fm.baseline_model(
            basis="poly", degree=2, sframe=sf,
            nuisance_list=[df1, df2],
        )
    X = np.asarray(bm.design_matrix)
    # 4 (poly: 2 cols per run × 2 runs) + 2 (block intercepts)
    # + 2 (per-run trans_x) = 8 columns.
    assert X.shape == (n1 + n2, 8), f"unexpected baseline shape {X.shape}"
    # Per-run nuisance is block-diagonal: the run-1 nuisance column
    # is zero in the run-2 segment, and vice versa.
    nuis_cols = X[:, -2:]
    assert np.allclose(nuis_cols[:n1, 1], 0.0), (
        "run-2 trans_x should be zero in run 1 segment"
    )
    assert np.allclose(nuis_cols[n1:, 0], 0.0), (
        "run-1 trans_x should be zero in run 2 segment"
    )
