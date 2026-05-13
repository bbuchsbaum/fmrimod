"""Tests for the public replay convenience facade."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.dataset import FmriDataset
from fmrimod.glm.replay import (
    ContrastDelta,
    ReplayContractError,
    ReplayResult,
    replay,
    replay_fits,
)
from fmrimod.spec import Spec, SpecDiff, hrf


def _dataset() -> FmriDataset:
    rng = np.random.default_rng(77)
    events = pd.DataFrame(
        {
            "onset": [8.0, 28.0, 48.0, 68.0],
            "duration": [2.0, 2.0, 2.0, 2.0],
            "trial_type": ["A", "B", "A", "B"],
            "run": [1, 1, 1, 1],
        }
    )
    return fm.fmri_dataset(
        rng.standard_normal((80, 4)).astype(np.float64),
        tr=2.0,
        events=events,
    )


def test_public_replay_import_boundary() -> None:
    assert fm.replay is replay
    assert fm.replay_fits is replay_fits
    assert fm.ReplayResult is ReplayResult
    assert fm.ContrastDelta is ContrastDelta
    assert fm.ReplayContractError is ReplayContractError


def test_replay_facade_fits_both_specs_and_delegates_to_replay_fits(monkeypatch) -> None:
    replay_mod = importlib.import_module("fmrimod.glm.replay")
    captured = {}

    def fake_replay_fits(fit_a, fit_b, *, named_contrasts=None):
        captured["fit_a"] = fit_a
        captured["fit_b"] = fit_b
        captured["named_contrasts"] = named_contrasts
        return ReplayResult(
            diff=SpecDiff(),
            fit_a=fit_a,
            fit_b=fit_b,
            contrast_deltas=(),
        )

    monkeypatch.setattr(replay_mod, "replay_fits", fake_replay_fits)
    spec_a = Spec() + hrf("trial_type", norm="spm")
    spec_b = Spec() + hrf("trial_type", norm="unit_peak")

    result = replay(spec_a, spec_b, _dataset(), named_contrasts=("cond",))

    assert isinstance(result, ReplayResult)
    assert captured["fit_a"].provenance is not None
    assert captured["fit_b"].provenance is not None
    assert captured["named_contrasts"] == ("cond",)
