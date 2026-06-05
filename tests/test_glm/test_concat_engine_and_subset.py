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


def test_concat_engine_honors_dataset_censor() -> None:
    """The concat engine row-deletes the censor mask before solving."""
    rng = np.random.default_rng(0)
    TR = 2.0
    N = 80
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 96.0, 6),
            "duration": 0.0,
            "trial_type": ["A", "B"] * 3,
            "run": 1,
        }
    )
    Y = rng.normal(size=(N, 8))
    censor = np.zeros(N, dtype=bool)
    censor[3:8] = True  # drop 5 frames mid-design
    ds = fm.fmri_dataset(Y, tr=TR, events=events, censor=censor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_concat = fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm") + intercept(per="run"),
            ds,
            engine="concat",
        )
        fit_runwise = fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm") + intercept(per="run"),
            ds,
            engine="runwise",
        )
    # dfres should reflect (n - censored) - rank, not n - rank.
    n_kept = int((~censor).sum())
    rank = int(fit_concat.condition_report().runs[0].rank)
    assert fit_concat.residual_df == pytest.approx(float(n_kept - rank))
    # Betas should match the runwise engine which already honors censor.
    np.testing.assert_allclose(
        fit_concat.betas, fit_runwise.betas, atol=1e-10
    )


def test_matrix_dataset_accepts_flat_censor_and_splits_by_run_length() -> None:
    """A flat censor passed to ``matrix_dataset`` gets per-run-split."""
    from fmrimod.dataset.constructors import matrix_dataset

    rng = np.random.default_rng(0)
    N_per_run = 40
    N_RUNS = 2
    events = pd.DataFrame(
        {
            "onset": np.concatenate([
                np.linspace(8.0, 56.0, 4),
                np.linspace(8.0, 56.0, 4),
            ]),
            "duration": 0.0,
            "trial_type": ["A", "B"] * 4,
            "run": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    Y = rng.normal(size=(N_per_run * N_RUNS, 4))
    censor = np.zeros(N_per_run * N_RUNS, dtype=bool)
    censor[5:8] = True  # 3 frames in run 1
    censor[50:52] = True  # 2 frames in run 2
    ds = matrix_dataset(
        Y, tr=2.0, run_length=N_per_run, event_table=events, censor=censor
    )
    assert ds.censor is not None
    assert len(ds.censor) == 2
    assert int(ds.censor[0].sum()) == 3
    assert int(ds.censor[1].sum()) == 2


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


# -- 4. Concat engine guards for unsupported compositions --------------------
#
# ``fit_concat`` returns ``residuals=None`` and a single-element
# ``run_X=[X]``. Both the AR integration path and the robust (IRLS) refit
# path read *per-run* residuals via ``initial_fit["residuals"][r]``, so
# composing either with ``engine="concat"`` would crash with a
# ``TypeError: 'NoneType' object is not subscriptable`` deep inside the
# refit. ``fmri_lm`` detects both combinations up front and raises a clear
# ``NotImplementedError`` instead.

def test_concat_engine_rejects_robust_refit() -> None:
    """robust + concat raises a clear unsupported-composition error."""
    from fmrimod.model.config import FmriLmConfig, RobustOptions

    _events, ds = _two_run_events_and_dataset()
    spec = hrf("trial_type", basis="spm", norm="spm") + intercept(per="run")
    config = FmriLmConfig(robust=RobustOptions(type="huber"))
    with pytest.raises(NotImplementedError, match="robust.*concat|concat.*robust"):
        fm.fmri_lm(spec, ds, config=config, engine="concat")


def test_concat_engine_rejects_ar_prewhitening() -> None:
    """AR + concat raises a clear unsupported-composition error."""
    _events, ds = _two_run_events_and_dataset()
    spec = hrf("trial_type", basis="spm", norm="spm") + intercept(per="run")
    with pytest.raises(NotImplementedError, match="AR.*concat|concat.*AR"):
        fm.fmri_lm(spec, ds, ar="ar1", engine="concat")


# -- 5. Concat engine honors configured preprocessing ------------------------
#
# The runwise path applies volume weights and soft-subspace projection via
# ``_prepare_run_matrices``. The concat path used to apply only censoring
# and then solve, silently dropping those options. Both now flow through the
# same preprocessing pipeline, so concat estimates match the runwise engine
# (and differ from the unweighted/unprojected concat solve).

def test_concat_engine_applies_volume_weights() -> None:
    from fmrimod.model.config import FmriLmConfig, VolumeWeightOptions

    _events, ds = _two_run_events_and_dataset(seed=1)
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + intercept(per="run")
    )
    # Non-trivial weights over all concatenated rows (2 runs x 80).
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.2, 1.0, size=160)
    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_concat_w = fm.fmri_lm(spec, ds, config=config, engine="concat")
        fit_runwise_w = fm.fmri_lm(spec, ds, config=config, engine="runwise")
        fit_concat_unw = fm.fmri_lm(spec, ds, engine="concat")
    # Weights are now honored: concat matches the runwise weighted fit ...
    np.testing.assert_allclose(
        fit_concat_w.betas, fit_runwise_w.betas, atol=1e-10
    )
    # ... and the weighting actually changed the solution (no silent drop).
    assert not np.allclose(fit_concat_w.betas, fit_concat_unw.betas, atol=1e-6)


def test_concat_engine_applies_soft_subspace_projection() -> None:
    from fmrimod.glm.preprocess import soft_subspace_projection
    from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
    from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions

    _events, ds = _two_run_events_and_dataset(seed=2)
    spec = (
        hrf("trial_type", "run_label", basis="spm", norm="spm")
        + intercept(per="run")
    )
    # Nuisance regressors defined over all concatenated rows.
    rng = np.random.default_rng(11)
    nuisance = rng.normal(size=(160, 3))
    config = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(
            enabled=True, nuisance_matrix=nuisance, lam=0.0
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_concat_p = fm.fmri_lm(spec, ds, config=config, engine="concat")
        fit_concat_unp = fm.fmri_lm(spec, ds, engine="concat")
    # The projection actually changed the solution (no silent drop).
    assert not np.allclose(fit_concat_p.betas, fit_concat_unp.betas, atol=1e-6)
    # The concat engine routes the *whole stacked system* through the same
    # soft-subspace helper the runwise path uses, so it equals a manual
    # full-system residualization + OLS. (This is distinct from the runwise
    # engine, which residualizes each run against its own nuisance slice.)
    X = np.asarray(fit_concat_p.model.design_matrix_array(run=None), dtype=float)
    Y = np.asarray(ds.get_data_matrix(), dtype=float)
    Xp, Yp = soft_subspace_projection(X, Y, nuisance, 0.0)
    proj = fast_preproject(Xp)
    manual = fast_lm_matrix(Xp, Yp, proj)
    np.testing.assert_allclose(fit_concat_p.betas, manual.betas, atol=1e-10)
