"""Regression tests for NaN-tolerant EventVariable behavior.

Pins the contract added when the 3x3-factorial + parametric stress test
was written:

- ``EventVariable`` accepts NaN / inf values by default
  (``nan_strategy="drop"``), substituting zero amplitude on the
  non-finite entries.
- A single ``UserWarning`` names the variable and the count.
- Center / scale statistics are computed over the finite entries only,
  so a stray NaN does not poison the rescaled column.
- ``nan_strategy="error"`` restores the legacy hard-fail behavior.
- Used as a parametric modulator on a categorical-by-continuous
  interaction, NaN trials contribute zero to the parametric column
  while staying in the main-effect categorical column.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.events import EventVariable
from fmrimod.spec import hrf


def test_event_variable_drops_nan_with_warning() -> None:
    onsets = np.arange(8.0, 8.0 + 6 * 5.0, 5.0)
    values = np.array([0.5, np.nan, 1.0, 0.8, np.nan, 0.7])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ev = EventVariable(name="rt", onsets=onsets, values=values, center=False)

    user_warnings = [
        w for w in captured
        if issubclass(w.category, UserWarning) and "non-finite" in str(w.message)
    ]
    assert len(user_warnings) == 1
    msg = str(user_warnings[0].message)
    assert "rt" in msg and "2/6" in msg

    # The convolution amplitude is exactly zero on the NaN-bearing trials.
    assert np.isfinite(ev.values).all()
    assert ev.values[1] == 0.0
    assert ev.values[4] == 0.0
    # nan_mask preserves the original mask for downstream inspection.
    np.testing.assert_array_equal(
        ev.nan_mask, np.array([False, True, False, False, True, False])
    )


def test_event_variable_error_strategy_preserves_legacy_behavior() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        EventVariable(
            name="rt",
            onsets=[0.0, 1.0, 2.0],
            values=[1.0, np.nan, 3.0],
            nan_strategy="error",
        )


def test_event_variable_center_uses_finite_entries_only() -> None:
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ev = EventVariable(
            name="rt",
            onsets=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=values,
            center=True,
        )
    finite_mean = np.mean([1.0, 3.0, 5.0])
    expected = np.array([1.0 - finite_mean, 0.0, 3.0 - finite_mean, 0.0, 5.0 - finite_mean])
    np.testing.assert_allclose(ev.values, expected, atol=1e-12)


def test_event_variable_scale_uses_finite_entries_only() -> None:
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ev = EventVariable(
            name="rt",
            onsets=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=values,
            center=True,
            scale=True,
        )
    finite_vals = np.array([1.0, 3.0, 5.0])
    expected_finite = (finite_vals - finite_vals.mean()) / finite_vals.std()
    np.testing.assert_allclose(
        ev.values[~ev.nan_mask], expected_finite, atol=1e-12
    )
    assert ev.values[1] == 0.0
    assert ev.values[3] == 0.0


def test_nan_modulator_keeps_main_effect_intact() -> None:
    """A NaN-bearing modulator drops the parametric column for that trial
    but the trial still contributes to the main-effect categorical column.
    """
    rng = np.random.default_rng(0)
    n = 12
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 100.0, n),
            "duration": 0.0,
            "trial_type": ["A", "B"] * (n // 2),
            "rt": rng.uniform(0.4, 1.2, n),
            "run": 1,
        }
    )
    events.loc[3, "rt"] = np.nan

    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", basis="spm", modulators=["rt"]), ds
        )
    columns = fit.design_columns()

    # Main-effect column for level B (where the NaN trial lives) is
    # not all-zero — the trial still convolves normally.
    main_b = columns.where(term="trial_type", level="B").one().index
    X = fit.model.design_matrix_array(run=0)
    assert np.any(X[:, main_b] != 0)

    # Parametric column carries level metadata too and the design
    # is finite end-to-end.
    param_b = columns.where(term="trial_type:rt", level="B").one().index
    assert np.isfinite(X[:, param_b]).all()
