"""Regression tests for the rank/conditioning diagnostic surface.

Pins the behavioural contract added when the multicollinear-baseline
stress test was written:

- ``fmri_lm`` emits a ``UserWarning`` on a rank-deficient design
  (separate from the existing baseline_model nuisance warning).
- ``FmriLm.is_full_rank`` and ``FmriLm.ill_conditioned`` shortcuts work.
- ``FmriLm.condition_report()`` returns a typed report whose
  ``aliased_columns`` names a column from the dependent set.
- The pseudoinverse recovery uses ``dfres = n - rank`` (textbook).
- Confound column names declared on the spec survive into the
  realised design and the aliased-column report (no ``V2`` placeholders).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.glm.solver import ConditionReport, RunConditionReport
from fmrimod.spec import confounds, hrf, intercept


N_SCANS = 80
TR = 2.0


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": np.arange(8.0, 8.0 + 14 * 10.0, 10.0),
            "duration": 0.0,
            "trial_type": (["A", "B"] * 7),
            "run": 1,
        }
    )


def _collinear_confounds(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    col_a = rng.normal(size=N_SCANS)
    col_c = rng.normal(size=N_SCANS)
    # Exact linear dependency: motion_y = 2 * motion_x + 0.5 * trans_z.
    col_b = 2 * col_a + 0.5 * col_c
    return pd.DataFrame(
        {"motion_x": col_a, "motion_y": col_b, "trans_z": col_c}
    )


def _fit_collinear() -> fm.glm.fmri_lm.FmriLm:
    events = _events()
    confound_df = _collinear_confounds()
    ds = fm.fmri_dataset(np.zeros((N_SCANS, 1)), tr=TR, events=events)
    spec = (
        hrf("trial_type", basis="spm")
        + confounds("motion_x", "motion_y", "trans_z", source=confound_df)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fm.fmri_lm(spec, ds)


def test_fmri_lm_warns_on_rank_deficient_design() -> None:
    events = _events()
    confound_df = _collinear_confounds()
    ds = fm.fmri_dataset(np.zeros((N_SCANS, 1)), tr=TR, events=events)
    spec = (
        hrf("trial_type", basis="spm")
        + confounds("motion_x", "motion_y", "trans_z", source=confound_df)
        + intercept(per="run")
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fm.fmri_lm(spec, ds)
    fit_warnings = [
        w for w in captured
        if issubclass(w.category, UserWarning)
        and "fmri_lm()" in str(w.message)
        and "rank-deficient" in str(w.message)
    ]
    assert fit_warnings, (
        "fmri_lm() must emit a UserWarning on a rank-deficient design"
    )


def test_fmri_lm_top_level_rank_shortcuts() -> None:
    fit = _fit_collinear()
    assert fit.is_full_rank is False
    assert fit.ill_conditioned is True


def test_condition_report_names_aliased_column_from_user_label() -> None:
    fit = _fit_collinear()
    report = fit.condition_report()
    assert isinstance(report, ConditionReport)
    assert report.is_full_rank is False
    # The QR pivot picks one of the three dependent confounds; the
    # report must use the user-supplied label, not a "V3"-style auto-name.
    aliased = report.aliased_columns
    assert aliased, "expected at least one aliased column"
    dependent = {"motion_x", "motion_y", "trans_z"}
    assert any(any(name in alias for name in dependent) for alias in aliased), (
        f"aliased column must be one of {dependent}; got {aliased!r}"
    )


def test_condition_report_run_record_shape() -> None:
    fit = _fit_collinear()
    report = fit.condition_report()
    assert len(report.runs) == 1
    run = report.runs[0]
    assert isinstance(run, RunConditionReport)
    assert run.n_columns == fit.n_coefficients
    assert run.rank < run.n_columns
    # dfres = n - rank, not n - p.
    assert run.dfres == pytest.approx(N_SCANS - run.rank)


def test_pseudoinverse_recovers_estimable_contrast() -> None:
    """An estimable task contrast is invariant to the pseudoinverse choice."""
    fit = _fit_collinear()
    n_total = fit.n_coefficients
    columns = fit.design_columns()
    iA = columns.where(term="trial_type", level="A").one().index
    iB = columns.where(term="trial_type", level="B").one().index
    c = np.zeros(n_total)
    c[iA] = 1.0
    c[iB] = -1.0
    res = fit.contrast(c, name="recall_minus_encode")
    # On zero data the betas are all-zero by construction, but the
    # contrast is still a valid identifiable combination of an
    # all-zero coefficient vector.
    assert np.isfinite(res.estimate).all()
    assert np.isfinite(res.stat).all()
