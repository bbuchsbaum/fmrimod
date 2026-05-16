"""Seed-based connectivity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.stats import seed_target_correlation, seed_target_lm


def test_seed_target_correlation_matches_manual_cleaned_signal_formula() -> None:
    clean = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 4.0],
            [3.0, 6.0, 3.0],
            [4.0, 8.0, 2.0],
            [5.0, 10.0, 1.0],
        ],
        dtype=np.float64,
    )

    out = seed_target_correlation(clean, seed=0, target_names=["positive", "negative"])

    seed = clean[:, 0]
    targets = clean[:, 1:]
    sc = seed - seed.mean()
    tc = targets - targets.mean(axis=0)
    expected = (sc @ tc) / (np.sqrt((sc**2).sum()) * np.sqrt((tc**2).sum(axis=0)))
    assert out["target"].tolist() == ["positive", "negative"]
    np.testing.assert_allclose(out["r"].to_numpy(), expected, atol=1e-12)
    assert out["n"].tolist() == [5, 5]
    assert out.loc[0, "fisher_z"] > 10
    assert out.loc[1, "fisher_z"] < -10
    np.testing.assert_allclose(out["z_variance"].to_numpy(), [0.5, 0.5])


def test_seed_target_correlation_preserves_dataframe_target_names() -> None:
    signals = pd.DataFrame(
        {
            "seed": [0.0, 1.0, 2.0, 3.0, 4.0],
            "roi_a": [0.0, 2.0, 4.0, 6.0, 8.0],
            "roi_b": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    out = seed_target_correlation(signals, seed="seed", targets=["roi_a", "roi_b"])

    assert out["target"].tolist() == ["roi_a", "roi_b"]
    np.testing.assert_allclose(out.loc[0, "r"], 1.0, atol=1e-12)


def test_seed_target_correlation_marks_zero_variance_targets_nan() -> None:
    clean = np.column_stack([
        np.arange(5, dtype=np.float64),
        np.ones(5, dtype=np.float64),
    ])

    out = seed_target_correlation(clean, seed=0, fisher=False)

    assert np.isnan(out.loc[0, "r"])
    assert list(out.columns) == ["target", "r", "n"]


def test_seed_target_lm_uses_fmri_lm_for_adjusted_seed_effect() -> None:
    n = 80
    t = np.linspace(-1.0, 1.0, n)
    confound = np.sin(np.linspace(0.0, np.pi * 2.0, n))
    seed = t + 0.4 * confound
    noise = np.cos(np.linspace(0.0, np.pi * 6.0, n))
    targets = np.column_stack([
        0.75 * seed + 2.0 * confound + 0.01 * noise,
        -0.25 * seed + 1.5 * confound - 0.01 * noise,
    ])
    signals = pd.DataFrame(
        {
            "seed_roi": seed,
            "target_a": targets[:, 0],
            "target_b": targets[:, 1],
        }
    )

    result = seed_target_lm(
        signals,
        seed="seed_roi",
        targets=["target_a", "target_b"],
        confounds=pd.DataFrame({"drift": confound}),
    )
    frame = result.to_frame()

    assert result.fit.n_coefficients == 3
    assert result.fit.design_columns().names == ("intercept", "drift", "seed")
    assert frame["target"].tolist() == ["target_a", "target_b"]
    np.testing.assert_allclose(frame["estimate"].to_numpy(), [0.75, -0.25], atol=0.01)
    assert np.all(frame["p"].to_numpy() < 1e-20)


def test_seed_target_lm_validates_confound_rows() -> None:
    clean = np.arange(20, dtype=np.float64).reshape(5, 4)

    with pytest.raises(ValueError, match="same number of rows"):
        seed_target_lm(clean, confounds=np.ones((4, 1)))
