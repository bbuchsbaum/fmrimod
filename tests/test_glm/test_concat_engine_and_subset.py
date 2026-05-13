"""Regression tests for the round-six pain-point fixes.

Four issues surfaced during the concatenated multi-run stress test:

1. ``subset=`` predicates on ``hrf(...)`` were accepted but silently
   ignored. Now: dict, predicate-string, and callable predicates all
   filter the events used for that term.
2. ``fmri_lm`` warned spuriously about per-run rank deficiency on
   orthogonal multi-run designs (each per-run sub-X is technically
   rank-deficient but the concatenated X is full rank). Now: the
   warning checks the concatenated rank and stays silent when only
   the per-run sub-blocks are deficient.
3. ``_pool_run_results`` raised ``RuntimeWarning: divide by zero``
   when zero-variance columns appeared in any run. Now: replaced
   ``np.where`` with ``np.divide(..., where=...)``.
4. ``fmri_lm`` had no single-concatenated-design strategy. Now:
   ``engine="concat"`` runs a single OLS on the stacked X / Y with
   textbook ``dfres = n - rank``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset.constructors import matrix_dataset
from fmrimod.spec import drift, hrf, intercept


def _two_run_events_and_dataset(seed: int = 0):
    """Return a 2-run dataset with a trivial 2-condition design."""
    rng = np.random.default_rng(seed)
    TR = 2.0
    N = 80
    rows = []
    for run_id in (1, 2):
        for k, onset in enumerate(np.linspace(8.0, 96.0, 6)):
            rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "trial_type": "A" if k % 2 == 0 else "B",
                    "run_label": f"run{run_id}",
                    "block": run_id,
                    "run": run_id,
                }
            )
    events = pd.DataFrame(rows)
    Y = rng.normal(size=(2 * N, 12))
    ds = matrix_dataset(Y, tr=TR, run_length=N, event_table=events)
    return events, ds


# -- 1. subset= predicate plumbing -------------------------------------------

def test_subset_dict_filters_events_for_one_term() -> None:
    rng = np.random.default_rng(0)
    n = 12
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 100.0, n),
            "duration": 0.0,
            "trial_type": "A",
            "block": [1] * 6 + [2] * 6,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", basis="spm", subset={"block": 1}), ds
        )
    X = fit.model.design_matrix_array(run=0)
    # Only the first 6 events (onsets 8-50 s, peaks 14-62 s) should
    # contribute; samples beyond ~75 s should be near zero.
    late_samples = X[35:, 0]  # t >= 71 s
    assert np.max(np.abs(late_samples)) < 0.05, (
        f"events past block 1 leaked into the regressor; max late "
        f"amplitude = {np.max(np.abs(late_samples)):.4g}"
    )


def test_subset_string_predicate_filters_events() -> None:
    n = 12
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 100.0, n),
            "duration": 0.0,
            "trial_type": "A",
            "block": [1] * 6 + [2] * 6,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_dict = fm.fmri_lm(
            hrf("trial_type", basis="spm", subset={"block": 1}), ds
        )
        fit_str = fm.fmri_lm(
            hrf("trial_type", basis="spm", subset="block == 1"), ds
        )
    np.testing.assert_allclose(
        fit_dict.model.design_matrix_array(run=0),
        fit_str.model.design_matrix_array(run=0),
        atol=1e-12,
    )


def test_subset_callable_predicate_filters_events() -> None:
    n = 12
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 100.0, n),
            "duration": 0.0,
            "trial_type": "A",
            "block": [1] * 6 + [2] * 6,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_dict = fm.fmri_lm(
            hrf("trial_type", basis="spm", subset={"block": 2}), ds
        )
        fit_call = fm.fmri_lm(
            hrf("trial_type", basis="spm", subset=lambda df: df["block"] == 2),
            ds,
        )
    np.testing.assert_allclose(
        fit_dict.model.design_matrix_array(run=0),
        fit_call.model.design_matrix_array(run=0),
        atol=1e-12,
    )


def test_subset_two_terms_produce_orthogonal_regressors() -> None:
    """Two subset-filtered terms with different ids span disjoint events."""
    n = 12
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 100.0, n),
            "duration": 0.0,
            "trial_type": "A",
            "block": [1] * 6 + [2] * 6,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    spec = (
        hrf("trial_type", basis="spm", subset={"block": 1}, id="b1")
        + hrf("trial_type", basis="spm", subset={"block": 2}, id="b2")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds)
    X = fit.model.design_matrix_array(run=0)
    task_b1_idx = fit.design_columns().where(term="b1").one().index
    task_b2_idx = fit.design_columns().where(term="b2").one().index
    b1_col = X[:, task_b1_idx]
    b2_col = X[:, task_b2_idx]
    # Different events in each block → genuinely different regressors
    # (the HRF tails overlap, so the columns aren't orthogonal, but the
    # peak amplitudes should land in different time windows).
    peak_b1 = int(np.argmax(b1_col))
    peak_b2 = int(np.argmax(b2_col))
    assert peak_b1 < peak_b2 - 10, (
        f"subset-filtered terms should peak in different windows; "
        f"b1 peak at sample {peak_b1}, b2 peak at sample {peak_b2}"
    )
    assert not np.allclose(b1_col, b2_col), (
        "subset-filtered terms should not produce identical regressors"
    )


def test_subset_empty_match_raises_clearly() -> None:
    events = pd.DataFrame(
        {
            "onset": [10.0, 20.0],
            "duration": 0.0,
            "trial_type": "A",
            "block": [1, 1],
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with pytest.raises(ValueError, match="matched zero events"):
        fm.fmri_lm(hrf("trial_type", basis="spm", subset={"block": 999}), ds)


# -- 2. Spurious multi-run rank warning ---------------------------------------

def test_multirun_orthogonal_design_does_not_warn() -> None:
    """A full-rank concatenated design with rank-deficient per-run blocks
    should not emit the rank-deficient warning."""
    events, ds = _two_run_events_and_dataset()
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fm.fmri_lm(spec, ds)
    rank_warnings = [
        w for w in captured
        if issubclass(w.category, UserWarning)
        and "fmri_lm()" in str(w.message)
        and "rank-deficient" in str(w.message)
    ]
    assert not rank_warnings, (
        "did not expect a rank-deficient warning on a multi-run "
        "orthogonal design whose concatenated X is full rank; got: "
        f"{[str(w.message) for w in rank_warnings]}"
    )


def test_multirun_orthogonal_design_does_not_divide_by_zero() -> None:
    """Per-run pooling on orthogonal multi-run designs no longer triggers
    RuntimeWarning: divide by zero."""
    events, ds = _two_run_events_and_dataset()
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + intercept(per="run")
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fm.fmri_lm(spec, ds)
    runtime = [
        w for w in captured
        if issubclass(w.category, RuntimeWarning)
        and "divide by zero" in str(w.message)
    ]
    assert not runtime, (
        "did not expect divide-by-zero RuntimeWarning from the pool "
        "step on orthogonal multi-run designs"
    )


# -- 3. Concat engine --------------------------------------------------------

def test_concat_engine_matches_runwise_betas() -> None:
    events, ds = _two_run_events_and_dataset()
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_runwise = fm.fmri_lm(spec, ds)
        fit_concat = fm.fmri_lm(spec, ds, engine="concat")
    # Identical betas (~1e-12 on this orthogonal design).
    np.testing.assert_allclose(
        fit_concat.betas, fit_runwise.betas, atol=1e-10
    )


def test_concat_engine_uses_textbook_dfres() -> None:
    events, ds = _two_run_events_and_dataset()
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")
    X = fit.model.design_matrix_array(run=None)
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    assert fit.residual_df == pytest.approx(n - rank)
    assert fit.is_full_rank is True


def test_concat_engine_single_run_works() -> None:
    """The concat engine is also valid on single-run datasets."""
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 96.0, 6),
            "duration": 0.0,
            "trial_type": ["A", "B"] * 3,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 4)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm"), ds, engine="concat"
        )
    assert fit.betas.shape == (3, 4)
    assert fit.is_full_rank is True
