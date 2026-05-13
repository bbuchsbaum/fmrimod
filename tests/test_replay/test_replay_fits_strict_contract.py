"""Contract tests for replay_fits strict comparison semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset import FmriDataset
from fmrimod.glm.replay import ReplayContractError, replay_fits
from fmrimod.spec import hrf


def _dataset() -> FmriDataset:
    rng = np.random.default_rng(20260513)
    events = pd.DataFrame(
        {
            "onset": [8.0, 28.0, 48.0, 68.0],
            "duration": [2.0, 2.0, 2.0, 2.0],
            "trial_type": ["A", "B", "A", "B"],
            "run": [1, 1, 1, 1],
        }
    )
    return fm.fmri_dataset(
        rng.standard_normal((80, 5)).astype(np.float64),
        tr=2.0,
        events=events,
    )


def _fit() -> fm.glm.FmriLm:
    fit = fm.fmri_lm(hrf("trial_type"), _dataset())
    weights = np.zeros(fit.n_coefficients, dtype=np.float64)
    weights[0] = 1.0
    fit.contrast(weights, name="common")
    fit.contrast(weights, name="only_a")
    return fit


def test_replay_fits_defaults_to_compatible_intersection_and_reports_asymmetry() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_b.contrasts.pop("only_a")
    weights = np.zeros(fit_b.n_coefficients, dtype=np.float64)
    weights[0] = 1.0
    fit_b.contrast(weights, name="only_b")

    result = replay_fits(fit_a, fit_b)

    assert [delta.name for delta in result.contrast_deltas] == ["common"]
    assert result.dropped_from_a == ("only_a",)
    assert result.dropped_from_b == ("only_b",)


def test_replay_fits_rejects_missing_explicit_named_contrast() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_b.contrasts.pop("only_a")

    with pytest.raises(ReplayContractError, match="absent"):
        replay_fits(fit_a, fit_b, named_contrasts=("only_a",))


def test_replay_fits_rejects_empty_default_intersection() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_a.contrasts.clear()
    fit_b.contrasts.clear()

    with pytest.raises(ReplayContractError, match="no compatible"):
        replay_fits(fit_a, fit_b)


def test_replay_fits_rejects_missing_fit_provenance() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_b.provenance = None

    with pytest.raises(ReplayContractError, match="missing FitProvenance"):
        replay_fits(fit_a, fit_b, named_contrasts=("common",))


def test_replay_fits_reports_max_and_median_stat_delta() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_a.contrasts["common"].stat = np.array([0.0, 0.0, 0.0, 0.0])
    fit_b.contrasts["common"].stat = np.array([1.0, 2.0, 100.0, 4.0])

    result = replay_fits(fit_a, fit_b, named_contrasts=("common",))
    delta = result.contrast_deltas[0]

    assert delta.max_abs_delta == pytest.approx(100.0)
    assert delta.median_abs_delta == pytest.approx(3.0)
    assert delta.value_a == pytest.approx(0.0)
    assert delta.value_b == pytest.approx(3.0)
    assert delta.df_match is True
    assert delta.stat_type == "t"


def test_replay_fits_rejects_t_df_mismatch() -> None:
    fit_a = _fit()
    fit_b = _fit()
    fit_b.contrasts["common"].df = fit_a.contrasts["common"].df + 1.0

    with pytest.raises(ReplayContractError, match="degrees of freedom"):
        replay_fits(fit_a, fit_b, named_contrasts=("common",))


def test_replay_fits_rejects_f_df_mismatch() -> None:
    fit_a = _fit()
    fit_b = _fit()
    weights = np.zeros((2, fit_a.n_coefficients), dtype=np.float64)
    weights[0, 0] = 1.0
    weights[1, min(1, fit_a.n_coefficients - 1)] = 1.0
    fit_a.contrast(weights, name="omnibus")
    fit_b.contrast(weights, name="omnibus")
    fit_b.contrasts["omnibus"].df = (2.0, fit_a.residual_df + 1.0)

    with pytest.raises(ReplayContractError, match="degrees of freedom"):
        replay_fits(fit_a, fit_b, named_contrasts=("omnibus",))
