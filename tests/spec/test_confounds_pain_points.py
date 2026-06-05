"""Regression tests pinning the typed ``confounds(...)`` API contract.

History: when the realistic-confounds parity workflow was wired in
early 2026 it surfaced three ergonomic gaps in the typed Spec layer
for nuisance regressors. Each gap was fixed in the same change that
landed this regression suite; the tests below pin the *current
(fixed) state* so a future change can't silently regress.

1. **Distinct ``role="confound"``** — confound columns no longer
   share ``role="baseline"`` with intercept and drift. Filtering
   them from the column registry uses typed lookup
   (``cols.where(role="confound")``), not name-suffix parsing.
2. **User-visible column name addressable as ``term``** — the
   prefixed realised column name (``"nuis_runK_<name>"``) is now
   parsed during colmap construction and the user's original
   DataFrame column name is exposed as ``DesignColumn.term``, so
   ``cols.where(term="trans_x")`` resolves directly.
3. **Multi-run confounds via the typed spec** — ``confounds(
   source=df)`` accepts both a single DataFrame (split row-wise
   along the dataset's block boundaries when the design has
   multiple runs) and a per-run sequence of DataFrames. Both
   shapes produce the same block-diagonal nuisance structure.
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


# -- Pain point 1 (fixed): distinct role="confound" ------------------------


def test_confound_columns_carry_distinct_confound_role() -> None:
    """``cols.where(role='confound')`` returns the confound columns.

    Before the fix, all baseline-source columns (intercept, drift,
    confound) shared ``role='baseline'`` and users had to suffix-match
    on the column name. The typed lookup story is now uniform with
    task columns (``cols.where(role='task')``).
    """
    fit = _single_run_fit()
    cols = fit.design_columns()
    confound_cols = list(cols.where(role="confound").columns)
    assert len(confound_cols) == 2, (
        f"expected 2 confound columns, got {len(confound_cols)}: "
        f"{[(c.index, c.name, c.role) for c in confound_cols]}"
    )
    for c in confound_cols:
        assert c.role == "confound"
        # The other baseline-source roles (drift / intercept) must NOT
        # match the confound lookup.
        assert c.role != "drift"
        assert c.role != "intercept"
        assert c.role != "baseline"


def test_drift_and_intercept_carry_their_own_distinct_roles() -> None:
    """Drift and intercept columns are now distinguishable from confounds.

    The baseline-source roles (drift, intercept, confound) are each
    filterable by their own ``role`` value, so a future colmap
    refactor can't silently collapse them back into a single
    ``baseline`` lump. The exact intercept/drift assignment depends
    on which spec terms the user supplied (an ``intercept(per="run")``
    without an explicit ``drift(...)`` term lives in baseline_model's
    drift slot with ``basis="constant"``); the canonical example we
    pin here adds an explicit poly drift so both roles appear.
    """
    rng = np.random.default_rng(0)
    events = pd.DataFrame({
        "onset": np.linspace(8.0, 96.0, 8),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 4,
        "run": 1,
    })
    conf_df = pd.DataFrame({"trans_x": rng.normal(size=80)})
    ds = fm.fmri_dataset(
        np.zeros((80, 4)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    from fmrimod.spec import drift as drift_term
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type")
            + confounds("trans_x", source=conf_df)
            + drift_term("poly", degree=2)
            + intercept(per="run"),
            ds,
        )
    role_counts: dict[str, int] = {}
    for c in fit.design_columns().columns:
        role_counts[c.role] = role_counts.get(c.role, 0) + 1
    assert role_counts.get("task", 0) == 2  # A, B
    assert role_counts.get("drift", 0) == 2  # poly1, poly2
    assert role_counts.get("intercept", 0) == 1  # constant
    assert role_counts.get("confound", 0) == 1  # trans_x


# -- Pain point 2 (fixed): user name is the term ----------------------------


def test_confound_user_name_resolves_via_typed_term_lookup() -> None:
    """``cols.where(term='trans_x')`` returns the trans_x confound directly.

    Before the fix, the realised column name carried a ``"nuis_runN_"``
    prefix and the user's original DataFrame column name was not
    addressable via the typed lookup; users had to filter by name
    suffix. The colmap now parses the prefix and exposes the user-
    visible name as ``DesignColumn.term``.
    """
    fit = _single_run_fit()
    cols = fit.design_columns()
    trans_x_matches = list(cols.where(term="trans_x").columns)
    assert len(trans_x_matches) == 1, (
        f"expected exactly one 'trans_x' column; got "
        f"{[(c.index, c.term, c.name) for c in trans_x_matches]}"
    )
    assert trans_x_matches[0].role == "confound"
    assert "trans_x" in (trans_x_matches[0].name or "")

    rot_x_matches = list(cols.where(term="rot_x").columns)
    assert len(rot_x_matches) == 1


# -- Pain point 3 (fixed): multi-run confounds via the typed spec ----------


def test_typed_confounds_single_df_splits_along_block_boundaries() -> None:
    """A single concatenated DataFrame source splits row-wise on multi-run."""
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
        "fd":      rng.uniform(size=n1 + n2),
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
        + confounds("trans_x", "fd", source=conf_df)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")
    X = fit.model.design_matrix_array(run=None)
    cols = fit.design_columns()
    # Per-run confound columns surface separately (run 1 and run 2 both
    # carry a `trans_x` entry).
    trans_x_cols = list(cols.where(term="trans_x").columns)
    assert len(trans_x_cols) == 2, (
        f"expected one trans_x col per run; got {trans_x_cols}"
    )
    # Block-diagonal structure: run 1's trans_x is zero in run 2 and
    # vice versa.
    run1_col = X[:, trans_x_cols[0].index]
    run2_col = X[:, trans_x_cols[1].index]
    assert np.allclose(run1_col[n1:], 0.0)
    assert np.allclose(run2_col[:n1], 0.0)


def test_typed_confounds_per_run_list_produces_same_design() -> None:
    """A per-run sequence of DataFrames produces the same design as a single concat."""
    rng = np.random.default_rng(0)
    n1, n2 = 60, 60
    events = pd.DataFrame({
        "onset": [10.0, 30.0, 10.0, 30.0],
        "duration": 0.0,
        "trial_type": ["A", "B", "A", "B"],
        "run": [1, 1, 2, 2],
    })
    df_run1 = pd.DataFrame({
        "trans_x": rng.normal(size=n1), "fd": rng.uniform(size=n1),
    })
    df_run2 = pd.DataFrame({
        "trans_x": rng.normal(size=n2), "fd": rng.uniform(size=n2),
    })
    ds = matrix_dataset(
        np.zeros((n1 + n2, 1)),
        tr=2.0,
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    spec_list = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds("trans_x", "fd", source=[df_run1, df_run2])
        + intercept(per="run")
    )
    spec_concat = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds(
            "trans_x", "fd",
            source=pd.concat([df_run1, df_run2], ignore_index=True),
        )
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_list = fm.fmri_lm(spec_list, ds, engine="concat")
        fit_concat = fm.fmri_lm(spec_concat, ds, engine="concat")
    np.testing.assert_array_equal(
        fit_list.model.design_matrix_array(run=None),
        fit_concat.model.design_matrix_array(run=None),
    )


def test_typed_confounds_rejects_mismatched_row_count() -> None:
    """A single-DataFrame source whose row count != sum(blocklens) raises."""
    rng = np.random.default_rng(0)
    n1, n2 = 60, 60
    events = pd.DataFrame({
        "onset": [10.0, 10.0], "duration": 0.0,
        "trial_type": ["A", "A"], "run": [1, 2],
    })
    bad_df = pd.DataFrame({"trans_x": rng.normal(size=80)})  # wrong length
    ds = matrix_dataset(
        np.zeros((n1 + n2, 1)),
        tr=2.0,
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds("trans_x", source=bad_df)
        + intercept(per="run")
    )
    with pytest.raises(ValueError, match="rows"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, engine="concat")


def test_typed_confounds_rejects_wrong_length_per_run_list() -> None:
    """A per-run sequence whose length != n_runs raises."""
    rng = np.random.default_rng(0)
    n1, n2 = 60, 60
    events = pd.DataFrame({
        "onset": [10.0, 10.0], "duration": 0.0,
        "trial_type": ["A", "A"], "run": [1, 2],
    })
    df_one = pd.DataFrame({"trans_x": rng.normal(size=n1)})  # only 1 run
    ds = matrix_dataset(
        np.zeros((n1 + n2, 1)),
        tr=2.0,
        run_length=[n1, n2],
        event_table=events,
        slice_timing_offset=0.0,
    )
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + confounds("trans_x", source=[df_one])
        + intercept(per="run")
    )
    with pytest.raises(ValueError, match="DataFrames"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, engine="concat")


def test_baseline_model_supports_multirun_nuisance_list_directly() -> None:
    """The underlying ``baseline_model(nuisance_list=...)`` path still works.

    Both the new typed-spec path AND the existing direct
    ``baseline_model`` API now produce per-run block-diagonal
    nuisance structure. Pinning the latter ensures the typed fix
    did not break the existing direct API surface.
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
    nuis_cols = X[:, -2:]
    assert np.allclose(nuis_cols[:n1, 1], 0.0)
    assert np.allclose(nuis_cols[n1:, 0], 0.0)
