"""Tests for simulate_simple_dataset helper."""

import numpy as np
import pytest

import fmrimod
from fmrimod.simulate import simulate_simple_dataset


def test_simulate_simple_dataset_shapes_and_keys():
    out = simulate_simple_dataset(ncond=3, nreps=5, tr=2.0, snr=0.8, seed=123)
    assert set(out.keys()) == {"clean", "noisy", "noise", "onsets", "conditions"}

    clean = out["clean"]
    noisy = out["noisy"]
    noise = out["noise"]
    onsets = out["onsets"]
    conditions = out["conditions"]

    assert clean.ndim == 2 and noisy.ndim == 2 and noise.ndim == 2
    assert clean.shape == noisy.shape
    assert clean.shape[1] == 1 + 3  # time + ncond signals
    assert noise.shape == clean[:, 1:].shape
    assert onsets.shape == (15,)
    assert len(conditions) == 15


def test_simulate_simple_dataset_is_reproducible_with_seed():
    a = simulate_simple_dataset(ncond=2, nreps=4, tr=1.5, snr=0.5, seed=42)
    b = simulate_simple_dataset(ncond=2, nreps=4, tr=1.5, snr=0.5, seed=42)
    np.testing.assert_allclose(a["clean"], b["clean"], atol=1e-12)
    np.testing.assert_allclose(a["noisy"], b["noisy"], atol=1e-12)
    np.testing.assert_allclose(a["noise"], b["noise"], atol=1e-12)
    np.testing.assert_allclose(a["onsets"], b["onsets"], atol=1e-12)
    assert a["conditions"] == b["conditions"]


def test_top_level_simulate_simple_dataset_wrapper():
    a = simulate_simple_dataset(ncond=2, nreps=3, seed=7)
    b = fmrimod.simulate_simple_dataset(ncond=2, nreps=3, seed=7)
    np.testing.assert_allclose(a["clean"], b["clean"], atol=1e-12)
    np.testing.assert_allclose(a["noisy"], b["noisy"], atol=1e-12)


def test_simulate_simple_dataset_validates_parameters():
    with pytest.raises(ValueError, match="ncond must be >= 1"):
        simulate_simple_dataset(ncond=0)
    with pytest.raises(ValueError, match="nreps must be >= 1"):
        simulate_simple_dataset(ncond=2, nreps=0)
    with pytest.raises(ValueError, match="tr must be > 0"):
        simulate_simple_dataset(ncond=2, tr=0.0)
    with pytest.raises(ValueError, match="snr must be > 0"):
        simulate_simple_dataset(ncond=2, snr=0.0)

