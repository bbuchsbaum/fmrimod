"""Tests for fmrireg compatibility simulation helpers."""

import numpy as np
import pytest

import fmrimod
from fmrimod.simulate import simulate_bold_signal, simulate_noise_vector


def test_simulate_bold_signal_shapes_and_keys():
    out = simulate_bold_signal(ncond=3, nreps=4, tr=2.0, seed=11)
    assert set(out.keys()) == {"onset", "condition", "mat"}
    onset = out["onset"]
    condition = out["condition"]
    mat = out["mat"]
    assert onset.shape == (12,)
    assert len(condition) == 12
    assert mat.ndim == 2
    assert mat.shape[1] == 4  # time + 3 conditions


def test_simulate_bold_signal_reproducible_with_seed():
    a = simulate_bold_signal(ncond=2, nreps=5, seed=3)
    b = simulate_bold_signal(ncond=2, nreps=5, seed=3)
    np.testing.assert_allclose(a["onset"], b["onset"], atol=1e-12)
    np.testing.assert_allclose(a["mat"], b["mat"], atol=1e-12)
    assert a["condition"] == b["condition"]


def test_simulate_noise_vector_reproducible_with_seed():
    a = simulate_noise_vector(n=200, tr=1.5, seed=9)
    b = simulate_noise_vector(n=200, tr=1.5, seed=9)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_simulate_noise_vector_physio_flag_changes_signal():
    with_phys = simulate_noise_vector(n=120, tr=2.0, physio=True, seed=8)
    no_phys = simulate_noise_vector(n=120, tr=2.0, physio=False, seed=8)
    assert with_phys.shape == (120,)
    assert no_phys.shape == (120,)
    assert not np.allclose(with_phys, no_phys)


def test_top_level_exports_work():
    a = fmrimod.simulate_bold_signal(ncond=2, nreps=3, seed=5)
    b = fmrimod.simulate_noise_vector(n=50, seed=5)
    assert a["mat"].shape[1] == 3
    assert b.shape == (50,)


def test_validation_errors():
    with pytest.raises(ValueError, match="ncond must be positive"):
        simulate_bold_signal(ncond=0)
    with pytest.raises(ValueError, match="Length of amps must equal ncond"):
        simulate_bold_signal(ncond=2, amps=[1.0])
    with pytest.raises(ValueError, match="isi must be length-2"):
        simulate_bold_signal(ncond=2, isi=(4.0, 4.0))
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_noise_vector(n=0)
    with pytest.raises(ValueError, match="sd must be positive"):
        simulate_noise_vector(n=10, sd=0.0)

