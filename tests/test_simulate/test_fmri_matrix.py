"""Tests for simulate_fmri_matrix parity helper."""

import numpy as np
import pytest

import fmrimod
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.simulate import simulate_fmri_matrix


def test_simulate_fmri_matrix_output_structure_and_shapes():
    out = simulate_fmri_matrix(
        n=3,
        total_time=80,
        TR=2.0,
        n_events=8,
        noise_type="white",
        random_seed=123,
    )
    assert set(out.keys()) == {"time_series", "ampmat", "durmat", "hrf_info", "noise_params"}
    assert isinstance(out["time_series"], FmriDataset)
    assert out["time_series"].get_all_data().shape[1] == 3
    assert out["ampmat"].shape == (8, 3)
    assert out["durmat"].shape == (8, 3)
    assert out["time_series"].event_table is not None
    assert {"run", "onset", "duration", "amplitude"}.issubset(out["time_series"].event_table.columns)


def test_simulate_fmri_matrix_reproducible_with_seed():
    a = simulate_fmri_matrix(
        n=2, total_time=60, TR=2.0, n_events=6, noise_type="ar1", random_seed=7
    )
    b = simulate_fmri_matrix(
        n=2, total_time=60, TR=2.0, n_events=6, noise_type="ar1", random_seed=7
    )
    np.testing.assert_allclose(a["ampmat"], b["ampmat"], atol=1e-12)
    np.testing.assert_allclose(a["durmat"], b["durmat"], atol=1e-12)
    np.testing.assert_allclose(
        a["time_series"].get_all_data(), b["time_series"].get_all_data(), atol=1e-12
    )


def test_simulate_fmri_matrix_accepts_explicit_onsets_and_single_trial_mode():
    out = simulate_fmri_matrix(
        n=1,
        total_time=50,
        TR=1.0,
        onsets=[5, 15, 25],
        n_events=10,  # ignored when onsets supplied
        single_trial=True,
        noise_type="none",
        random_seed=11,
    )
    assert out["ampmat"].shape[0] == 3
    assert out["durmat"].shape[0] == 3
    assert out["time_series"].event_table.shape[0] == 3
    assert out["hrf_info"]["single_trial"] is True


def test_top_level_simulate_fmri_matrix_wrapper():
    out = fmrimod.simulate_fmri_matrix(
        n=2,
        total_time=40,
        TR=2.0,
        n_events=4,
        noise_type="none",
        random_seed=1,
    )
    assert isinstance(out["time_series"], FmriDataset)
    assert out["time_series"].get_all_data().shape[1] == 2


def test_simulate_fmri_matrix_validation():
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_fmri_matrix(n=0)
    with pytest.raises(ValueError, match="total_time must be positive"):
        simulate_fmri_matrix(total_time=0)
    with pytest.raises(ValueError, match="TR must be positive"):
        simulate_fmri_matrix(TR=0)
    with pytest.raises(ValueError, match="No valid onsets generated"):
        simulate_fmri_matrix(onsets=[], random_seed=1)

