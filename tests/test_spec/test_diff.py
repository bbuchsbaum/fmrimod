"""Structural diff between two Spec trees.

The contract under test:
- Identical specs return an empty diff (``is_empty`` true, ``bool(diff)``
  false).
- A scalar field change shows up on exactly the changed term, with both
  old and new values present in the FieldDiff.
- Adding / removing / reordering event or baseline terms partitions
  correctly into ``added_*`` / ``removed_*`` / ``changed_*``.
- Matching prefers ``HrfTerm.id`` over ``(variables, hrf)`` so a
  renamed identifier is treated as remove+add rather than a confusing
  field-level diff.
- pandas DataFrame payloads on Confounds.source diff via
  ``DataFrame.equals`` rather than element-wise ``==``.
- Callable predicates compare by identity, so two distinct lambdas are
  always different (avoids both false-positives and false-negatives
  from arbitrary-callable equality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod
from fmrimod.spec import (
    FieldDiff,
    SpecDiff,
    TermDiff,
    confounds,
    drift,
    hrf,
    intercept,
    spec_diff,
)


# ---------------------------------------------------------------------------
# No-op diffs
# ---------------------------------------------------------------------------


def test_identical_specs_produce_empty_diff():
    left = hrf("trial_type") + drift("cosine", cutoff=128) + intercept()
    right = hrf("trial_type") + drift("cosine", cutoff=128) + intercept()
    diff = spec_diff(left, right)
    assert diff.is_empty
    assert not bool(diff)
    assert diff == SpecDiff()


def test_spec_method_matches_function_form():
    left = hrf("trial_type") + drift("cosine", cutoff=128)
    right = hrf("trial_type") + drift("cosine", cutoff=64)
    assert left.diff(right) == spec_diff(left, right)


def test_top_level_spec_diff_is_exposed():
    """spec_diff and the diff types are reachable from `fmrimod` itself."""
    assert fmrimod.spec_diff is spec_diff
    assert fmrimod.SpecDiff is SpecDiff
    assert fmrimod.TermDiff is TermDiff
    assert fmrimod.FieldDiff is FieldDiff


# ---------------------------------------------------------------------------
# Scalar field changes
# ---------------------------------------------------------------------------


def test_drift_cutoff_change_is_localised_to_one_field():
    left = hrf("trial_type") + drift("cosine", cutoff=128)
    right = hrf("trial_type") + drift("cosine", cutoff=64)
    diff = spec_diff(left, right)
    assert not diff.is_empty
    assert diff.added_events == ()
    assert diff.removed_events == ()
    assert diff.changed_events == ()
    assert len(diff.changed_baseline) == 1
    td = diff.changed_baseline[0]
    assert set(td.fields) == {"cutoff"}
    assert td.fields["cutoff"] == FieldDiff(left=128, right=64)


def test_intercept_per_change_diffs_only_the_per_field():
    left = hrf("trial_type") + intercept(per="run")
    right = hrf("trial_type") + intercept(per="global")
    diff = spec_diff(left, right)
    assert len(diff.changed_baseline) == 1
    assert set(diff.changed_baseline[0].fields) == {"per"}


def test_hrf_basis_change_diffs_only_the_hrf_field_when_variables_match():
    left = hrf("trial_type", basis="spm")
    right = hrf("trial_type", basis="spmg3")
    diff = spec_diff(left, right)
    # Same variables tuple matches the two events; only `hrf` differs.
    assert len(diff.changed_events) == 1
    assert set(diff.changed_events[0].fields) == {"hrf"}


# ---------------------------------------------------------------------------
# Adds and removes
# ---------------------------------------------------------------------------


def test_added_event_appears_in_added_events_bucket():
    left = hrf("trial_type") + drift("cosine", cutoff=128)
    right = (
        hrf("trial_type")
        + hrf("block")
        + drift("cosine", cutoff=128)
    )
    diff = spec_diff(left, right)
    assert len(diff.added_events) == 1
    assert diff.added_events[0].variables == ("block",)
    assert diff.removed_events == ()
    assert diff.changed_events == ()


def test_removed_baseline_term_appears_in_removed_baseline_bucket():
    left = hrf("trial_type") + drift("cosine", cutoff=128) + intercept()
    right = hrf("trial_type") + drift("cosine", cutoff=128)
    diff = spec_diff(left, right)
    assert len(diff.removed_baseline) == 1
    assert diff.removed_baseline[0].__class__.__name__ == "Intercept"


# ---------------------------------------------------------------------------
# Identity matching
# ---------------------------------------------------------------------------


def test_id_renaming_is_remove_plus_add_not_a_changed_event():
    """Changing only the explicit `id` should NOT field-diff to ``id``;
    the two terms cease to refer to the same logical entity."""
    left = hrf("trial_type", id="trials_a")
    right = hrf("trial_type", id="trials_b")
    diff = spec_diff(left, right)
    assert len(diff.removed_events) == 1
    assert len(diff.added_events) == 1
    assert diff.changed_events == ()


def test_id_match_overrides_variables_match():
    """When both sides set the same id, terms match even if their
    variables differ -- the user is saying 'this is the same term, I
    redefined it'."""
    left = hrf("trial_type", id="trials")
    right = hrf("block", id="trials")
    diff = spec_diff(left, right)
    assert diff.added_events == ()
    assert diff.removed_events == ()
    assert len(diff.changed_events) == 1
    assert set(diff.changed_events[0].fields) == {"variables"}


# ---------------------------------------------------------------------------
# DataFrame payloads
# ---------------------------------------------------------------------------


def test_confounds_source_compares_via_dataframe_equals():
    df_a = pd.DataFrame({"trans_x": [0.1, 0.2], "trans_y": [0.0, 0.0]})
    df_b = pd.DataFrame({"trans_x": [0.1, 0.2], "trans_y": [0.0, 0.0]})  # equal
    df_c = pd.DataFrame({"trans_x": [0.1, 0.3], "trans_y": [0.0, 0.0]})  # different
    left = confounds("trans_x", "trans_y", source=df_a)
    same = confounds("trans_x", "trans_y", source=df_b)
    different = confounds("trans_x", "trans_y", source=df_c)
    assert spec_diff(left, same).is_empty
    diff = spec_diff(left, different)
    assert len(diff.changed_baseline) == 1
    assert set(diff.changed_baseline[0].fields) == {"source"}


# ---------------------------------------------------------------------------
# Predicate / callable comparison
# ---------------------------------------------------------------------------


def test_distinct_callable_subsets_diff_as_different():
    """Two distinct lambdas are unequal under our 'callable -> identity'
    rule, even if they happen to do the same thing."""
    left = fmrimod.spec.hrf("trial_type", subset=lambda df: df["block"] == 1)
    right = fmrimod.spec.hrf("trial_type", subset=lambda df: df["block"] == 1)
    diff = spec_diff(left, right)
    assert len(diff.changed_events) == 1
    assert set(diff.changed_events[0].fields) == {"subset"}


def test_same_callable_instance_is_equal():
    """A shared callable reference on both sides is considered equal."""
    sub = lambda df: df["block"] == 1
    left = fmrimod.spec.hrf("trial_type", subset=sub)
    right = fmrimod.spec.hrf("trial_type", subset=sub)
    assert spec_diff(left, right).is_empty


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_summary_handles_empty_diff():
    left = hrf("trial_type")
    right = hrf("trial_type")
    assert spec_diff(left, right).summary() == "(no differences)"


def test_summary_mentions_changed_field_names():
    left = hrf("trial_type") + drift("cosine", cutoff=128)
    right = hrf("trial_type") + drift("cosine", cutoff=64)
    text = spec_diff(left, right).summary()
    assert "cutoff" in text
    assert "128" in text and "64" in text


def test_summary_mentions_added_and_removed_terms():
    left = hrf("trial_type") + intercept()
    right = hrf("trial_type") + hrf("block")
    text = spec_diff(left, right).summary()
    assert "added" in text
    assert "removed" in text
    assert "block" in text
