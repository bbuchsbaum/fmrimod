"""Regression tests for parametric-modulator handling.

Pins fmrimod's modulator semantics to the modern-correct behavior
(Mumford et al. 2015, *PLoS ONE*):

- ``hrf("trial_type", modulators=("rt", "accuracy"))`` expands to one
  unmodulated boxcar plus one per-modulator regressor — three task
  columns total for a single-condition design.
- Modulators are **not** orthogonalized by default. The order of
  modulator names does not affect the betas (modulo column
  reordering).
- Modulators are **not** auto-centered. The user must mean-center
  modulators in the events DataFrame before convolution, otherwise
  the unmodulated boxcar and the modulated regressors are
  highly collinear. Logged as a pain point for follow-up
  (a ``center_modulators=`` keyword would be a clean ergonomic
  win).
- The correct inferential question "do these modulators add
  variance beyond the unmodulated regressor?" is answered with a
  joint F-test over modulator columns, not by relying on
  orthogonalization to make individual betas look identifiable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.spec import drift, hrf, intercept


def _make_events(seed: int = 0, n: int = 12, centered: bool = True) -> pd.DataFrame:
    """Twelve single-condition trials with optional pre-centering."""
    rng = np.random.default_rng(seed)
    rt = rng.uniform(0.5, 1.5, n)
    acc = rng.uniform(0.0, 1.0, n)
    if centered:
        rt = rt - rt.mean()
        acc = acc - acc.mean()
    return pd.DataFrame({
        "onset": np.linspace(8.0, 96.0, n),
        "duration": 0.0,
        "trial_type": "A",
        "rt": rt,
        "accuracy": acc,
        "run": 1,
    })


def _fit_with_modulators(
    events: pd.DataFrame, modulators: tuple[str, ...]
) -> object:
    ds = fm.fmri_dataset(
        np.zeros((80, 1)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fm.fmri_lm(
            hrf("trial_type", modulators=modulators), ds
        )


def test_modulators_expand_to_unmod_plus_per_modulator() -> None:
    """``modulators=(...)`` gives one unmodulated boxcar plus one column per modulator."""
    events = _make_events(centered=True)
    fit = _fit_with_modulators(events, modulators=("rt", "accuracy"))
    cols = fit.design_columns()
    task_cols = [c for c in cols.columns if c.role == "task"]
    assert len(task_cols) == 3, (
        f"expected 1 unmodulated + 2 modulated = 3 task cols, got {len(task_cols)}"
    )
    # Each piece is addressable by typed level lookup.
    cols.where(term="trial_type", level="A").one()
    cols.where(term="trial_type:rt", level="A").one()
    cols.where(term="trial_type:accuracy", level="A").one()


def test_modulators_are_not_orthogonalized_by_default() -> None:
    """Reordering modulators does not change betas (modulo column reordering).

    If fmrimod were orthogonalizing sequentially against earlier
    modulators (the SPM12 default), the *first*-listed modulator
    would absorb the shared variance and the per-modulator betas
    would depend on the order. This test confirms fmrimod does the
    modern-correct thing: leaves modulators alone (no Gram-Schmidt).
    """
    events = _make_events(centered=True)
    fit_rt_first = _fit_with_modulators(events, modulators=("rt", "accuracy"))
    fit_acc_first = _fit_with_modulators(events, modulators=("accuracy", "rt"))

    X1 = fit_rt_first.model.design_matrix_array(run=0)
    X2 = fit_acc_first.model.design_matrix_array(run=0)

    # Extract per-modulator columns from each fit.
    rt_col_1 = X1[:, fit_rt_first.design_columns().where(
        term="trial_type:rt", level="A").one().index]
    rt_col_2 = X2[:, fit_acc_first.design_columns().where(
        term="trial_type:rt", level="A").one().index]
    acc_col_1 = X1[:, fit_rt_first.design_columns().where(
        term="trial_type:accuracy", level="A").one().index]
    acc_col_2 = X2[:, fit_acc_first.design_columns().where(
        term="trial_type:accuracy", level="A").one().index]

    np.testing.assert_allclose(rt_col_1, rt_col_2, atol=1e-12, err_msg=(
        "rt modulator column changed under reordering — implies sequential "
        "orthogonalization, which fmrimod should NOT be doing by default"
    ))
    np.testing.assert_allclose(acc_col_1, acc_col_2, atol=1e-12, err_msg=(
        "accuracy modulator column changed under reordering"
    ))


def test_centered_modulators_are_near_orthogonal_to_unmodulated() -> None:
    """With pre-centered modulators, unmod–modulator correlations < 0.1.

    This pins the empirical claim that mean-centering modulators in
    the events DataFrame produces the near-orthogonal design that
    makes per-modulator betas interpretable.
    """
    events = _make_events(centered=True)
    fit = _fit_with_modulators(events, modulators=("rt", "accuracy"))
    X = fit.model.design_matrix_array(run=0)
    task_idx = [c.index for c in fit.design_columns().columns if c.role == "task"]
    cm = np.corrcoef(X[:, task_idx].T)
    # Off-diagonal entries
    assert abs(cm[0, 1]) < 0.1, f"unmod ↔ rt corr = {cm[0,1]:.4f}"
    assert abs(cm[0, 2]) < 0.1, f"unmod ↔ accuracy corr = {cm[0,2]:.4f}"


def test_default_centers_modulators_even_without_pre_centering() -> None:
    """The typed-spec default ``center_modulators=True`` centers raw values.

    Pre-centering in the events DataFrame is no longer required —
    the typed spec defaults to ``center_modulators=True`` which
    consumes raw values and applies the centering at the input-
    variable level (pre-convolution). Even with raw uncentered RT
    and accuracy in the events DataFrame, the realised design has
    near-orthogonal unmodulated / modulated columns.
    """
    events_uncentered = _make_events(centered=False)
    fit = _fit_with_modulators(events_uncentered, modulators=("rt", "accuracy"))
    X = fit.model.design_matrix_array(run=0)
    task_idx = [c.index for c in fit.design_columns().columns if c.role == "task"]
    cm = np.corrcoef(X[:, task_idx].T)
    assert abs(cm[0, 1]) < 0.1, (
        f"default center_modulators=True should produce near-orthogonal "
        f"unmod ↔ rt columns even with uncentered events; got {cm[0, 1]:.4f}"
    )


def test_center_modulators_false_preserves_collinearity() -> None:
    """Explicit ``center_modulators=False`` opts out of pre-centering.

    Available for R-fmridesign exact-value parity or the rare case
    where the modulator's absolute scale carries the analytic
    meaning. With raw uncentered modulators the unmodulated boxcar
    and modulated regressors are strongly collinear, as expected.
    """
    events_uncentered = _make_events(centered=False)
    ds = fm.fmri_dataset(
        np.zeros((80, 1)), tr=2.0, events=events_uncentered, slice_timing_offset=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf(
                "trial_type",
                modulators=("rt", "accuracy"),
                center_modulators=False,
            ),
            ds,
        )
    X = fit.model.design_matrix_array(run=0)
    task_idx = [c.index for c in fit.design_columns().columns if c.role == "task"]
    cm = np.corrcoef(X[:, task_idx].T)
    assert abs(cm[0, 1]) > 0.5, (
        f"center_modulators=False should preserve the raw-amplitude "
        f"collinearity; got unmod ↔ rt corr = {cm[0, 1]:.4f}"
    )


def test_centering_is_pre_convolution_input_variable_operation() -> None:
    """Centering acts on raw modulator values, NOT on the convolved column.

    This pins fmrimod's consistent semantics across all centering
    surfaces (``EventVariable(center=True)``, ``Scale(center=True)``,
    typed-spec ``center_modulators=True``): the operation is on the
    *input variable* (the scalar modulator value per event) before
    that scalar becomes the boxcar amplitude. The convolved
    regressor's mean is the natural consequence of the convolution,
    not a separately-applied centering step.
    """
    events_uncentered = _make_events(centered=False, n=10)
    events_pre_centered = events_uncentered.copy()
    for col in ("rt", "accuracy"):
        events_pre_centered[col] -= events_pre_centered[col].mean()

    # Two ways of getting to the same place: default typed spec on raw
    # events, vs default typed spec on pre-centered events.
    fit_auto = _fit_with_modulators(events_uncentered, modulators=("rt",))
    fit_pre = _fit_with_modulators(events_pre_centered, modulators=("rt",))

    X_auto = fit_auto.model.design_matrix_array(run=0)
    X_pre = fit_pre.model.design_matrix_array(run=0)
    rt_idx_auto = fit_auto.design_columns().where(
        term="trial_type:rt", level="A"
    ).one().index
    rt_idx_pre = fit_pre.design_columns().where(
        term="trial_type:rt", level="A"
    ).one().index

    # If centering happens at the right (input-variable) stage, the
    # realised modulator columns are bitwise-equal regardless of
    # whether the user pre-centered or relied on the typed-spec default.
    np.testing.assert_allclose(
        X_auto[:, rt_idx_auto], X_pre[:, rt_idx_pre], atol=1e-12,
        err_msg=(
            "Auto-center and pre-center realised columns should be "
            "bitwise-equal (proves centering is on the raw input variable, "
            "not on the convolved regressor)"
        ),
    )


def test_joint_f_over_modulators_is_well_defined() -> None:
    """The 2-DF joint F-test over modulators is contrast-evaluable.

    Per Mumford et al., the modern-correct way to ask "do these
    modulators add variance beyond the unmodulated regressor?" is a
    nested-model joint F over the modulator columns, not implicit
    orthogonalization. This test exercises the typed contrast API
    on that F-test.
    """
    events = _make_events(centered=True, n=20)
    fit = _fit_with_modulators(events, modulators=("rt", "accuracy"))
    cols = fit.design_columns()
    rt_idx = cols.where(term="trial_type:rt", level="A").one().index
    acc_idx = cols.where(term="trial_type:accuracy", level="A").one().index
    n_cols = sum(1 for _ in cols.columns)
    c = np.zeros((2, n_cols), dtype=np.float64)
    c[0, rt_idx] = 1.0
    c[1, acc_idx] = 1.0
    result = fit.contrast(c, name="modulators_joint")
    assert result.stat.shape == (1,)
    assert np.isfinite(result.stat).all()
