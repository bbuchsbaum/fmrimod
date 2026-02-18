"""Tests for estimate_hrf compatibility utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import fmrimod as fm
from fmrimod.single import estimate_hrf
from fmrimod.single._types import VoxelHrfResult
from fmrimod.dataset.adapters.numpy_adapter import NumpyAdapter
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.sampling import SamplingFrame


def _synthetic_inputs(seed: int = 123):
    rng = np.random.default_rng(seed)
    t, n_trials, k, v = 140, 9, 3, 6
    basis = rng.standard_normal((24, k))
    x_trials = rng.standard_normal((t, n_trials * k))
    coeff = rng.standard_normal((k, v))

    # Match estimate_voxel_hrf aggregate design definition.
    a = np.zeros((t, k))
    for j in range(k):
        a[:, j] = x_trials[:, j::k].sum(axis=1)

    y = a @ coeff
    coeff_norm = coeff / np.linalg.norm(coeff, axis=0, keepdims=True)
    return y, x_trials, basis, coeff_norm


def test_estimate_hrf_coefficients_recover_normalized_weights():
    y, x_trials, basis, coeff_norm = _synthetic_inputs()
    out = estimate_hrf(y, x_trials, basis, K=3, output="coefficients")
    assert out.shape == coeff_norm.shape
    assert_allclose(out, coeff_norm, atol=1e-10)


def test_estimate_hrf_output_hrf_reconstructs_basis_projection():
    y, x_trials, basis, coeff_norm = _synthetic_inputs()
    out = estimate_hrf(y, x_trials, basis, K=3, output="hrf")
    expected = basis @ coeff_norm
    assert out.shape == expected.shape
    assert_allclose(out, expected, atol=1e-10)


def test_estimate_hrf_result_mode_returns_dataclass():
    y, x_trials, basis, _ = _synthetic_inputs()
    out = estimate_hrf(y, x_trials, basis, K=3, output="result")
    assert isinstance(out, VoxelHrfResult)
    assert out.coefficients.shape == (3, 6)


def test_estimate_hrf_requires_dataset_when_form_is_provided():
    y, x_trials, basis, _ = _synthetic_inputs()
    with pytest.raises(ValueError, match="requires both form and dataset"):
        estimate_hrf(y, x_trials, basis, form="onset ~ hrf(cond)")


def test_estimate_hrf_requires_core_matrix_inputs():
    with pytest.raises(ValueError, match="requires Y, X_trials, and basis"):
        estimate_hrf()


def test_top_level_estimate_hrf_export():
    y, x_trials, basis, _ = _synthetic_inputs()
    out = fm.estimate_hrf(y, x_trials, basis, K=3, output="coefficients")
    assert out.shape == (3, 6)


def _build_trial_basis_design(onsets, basis, n_timepoints, tr):
    onsets = np.asarray(onsets, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    L, K = basis.shape
    X = np.zeros((n_timepoints, len(onsets) * K), dtype=np.float64)
    onset_idx = np.rint(onsets / tr).astype(int)
    for i, idx0 in enumerate(onset_idx):
        m = min(L, n_timepoints - idx0)
        for k in range(K):
            X[idx0 : idx0 + m, i * K + k] = basis[:m, k]
    return X


def test_estimate_hrf_formula_dataset_mode_single_run():
    rng = np.random.default_rng(7)
    T, K, V = 80, 3, 4
    tr = 1.0
    onsets = np.array([5.0, 15.0, 25.0, 35.0, 45.0], dtype=np.float64)
    basis = rng.standard_normal((12, K))
    X_trials = _build_trial_basis_design(onsets, basis, T, tr)
    A = np.column_stack([X_trials[:, k::K].sum(axis=1) for k in range(K)])
    coeff = rng.standard_normal((K, V))
    Y = A @ coeff
    coeff_norm = coeff / np.linalg.norm(coeff, axis=0, keepdims=True)

    sf = SamplingFrame(blocklens=T, tr=tr)
    ds = FmriDataset(
        NumpyAdapter(Y, sf),
        event_table=pd.DataFrame(
            {
                "onset": onsets,
                "condition": ["A", "B", "A", "B", "A"],
            }
        ),
    )

    out = estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=ds,
        basis=basis,
        output="coefficients",
    )
    assert out.shape == (K, V)
    assert_allclose(out, coeff_norm, atol=1e-10)


def test_estimate_hrf_formula_dataset_mode_requires_single_run():
    rng = np.random.default_rng(9)
    T, K, V = 20, 2, 2
    basis = rng.standard_normal((6, K))
    Y0 = rng.standard_normal((T, V))
    Y1 = rng.standard_normal((T, V))
    sf = SamplingFrame(blocklens=[T, T], tr=[1.0, 1.0])
    ds = FmriDataset(
        NumpyAdapter([Y0, Y1], sf),
        event_table=pd.DataFrame({"onset": [2.0, 8.0], "condition": ["A", "B"]}),
    )
    with pytest.raises(NotImplementedError, match="single-run datasets only"):
        estimate_hrf(form="onset ~ hrf(condition)", dataset=ds, basis=basis)
