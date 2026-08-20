"""Red checks for the fmrireg 0.2.0 smooth-FIR ``estimate_hrf``.

Cheap pass disqualified: the old voxel-aggregate ``lstsq`` helper still
named ``estimate_hrf`` and wrapping ``estimate_voxel_hrf``. The formula
+ dataset path without a numeric 2-D basis must return ``HrfEstimate``
with a shared penalized solve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset import FmriDataset
from fmrimod.dataset.adapters import NumpyAdapter
from fmrimod.hrf.estimate import HrfEstimate, estimate_hrf
from fmrimod.sampling import SamplingFrame


def _condition_amplitudes(n_voxels: int) -> np.ndarray:
    amp_a = np.resize(np.array([1.0, 0.7, -0.6, 1.4]), n_voxels)
    return np.vstack([amp_a, 2.0 * amp_a])


def _make_spmg1_dataset(
    n: int = 220,
    n_voxels: int = 3,
    noise_sd: float = 0.025,
    seed: int = 180,
) -> tuple[FmriDataset, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    onsets = np.arange(8, n - 30, 10, dtype=np.float64)
    events = pd.DataFrame(
        {
            "onset": onsets,
            "condition": np.resize(["A", "B"], onsets.size),
            "run": 1,
        }
    )
    sframe = SamplingFrame(blocklens=n, tr=1.0)
    event = fm.event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=events,
        sampling_frame=sframe,
        durations=0.0,
        block="run",
    )
    x_event = np.asarray(event.design_matrix, dtype=np.float64)
    amplitudes = _condition_amplitudes(n_voxels)
    y = x_event @ amplitudes
    y = y + np.linspace(-0.3, 0.3, n_voxels)[np.newaxis, :]
    y = y + rng.normal(scale=noise_sd, size=y.shape)
    dataset = FmriDataset(NumpyAdapter(y, sframe), event_table=events)
    return dataset, amplitudes, events


def test_smooth_fir_recovers_known_hrf_on_near_noiseless_data():
    dataset, amplitudes, _events = _make_spmg1_dataset(noise_sd=0.0)
    sample_at = np.arange(0.0, 21.0)
    fit = estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=dataset,
        block="run",
        rsam=sample_at,
        k=8,
        lam=0.0,
    )
    assert isinstance(fit, HrfEstimate)
    assert fit.estimate.shape == (sample_at.size, 2, 3)
    assert fit.std_error.shape == fit.estimate.shape
    assert np.all(np.isfinite(fit.estimate))
    assert np.all(fit.std_error >= 0)
    assert not isinstance(fit, np.ndarray)

    truth_shape = np.asarray(fm.get_hrf("spmg1")(sample_at), dtype=np.float64).reshape(
        -1
    )
    for curve_i, row in fit.curve_info.iterrows():
        condition = str(row.condition).rsplit(".", 1)[-1]
        truth_row = 0 if condition.endswith("A") else 1
        for voxel in range(amplitudes.shape[1]):
            truth = truth_shape * amplitudes[truth_row, voxel]
            observed = fit.estimate[:, curve_i, voxel]
            corr = np.corrcoef(observed, truth)[0, 1]
            rmse = np.sqrt(np.mean((observed - truth) ** 2)) / np.max(np.abs(truth))
            assert corr > 0.9
            assert rmse < 0.35
    assert np.all(np.abs(fit.estimate[0]) < 1e-8)
    assert np.all(np.abs(fit.estimate[-1]) < 1e-8)


def test_gcv_lambda_beats_unpenalized_on_smooth_truth():
    dataset, amplitudes, _events = _make_spmg1_dataset(
        n=240, n_voxels=2, noise_sd=0.2, seed=19
    )
    sample_at = np.arange(0.0, 21.0)
    truth_shape = np.asarray(fm.get_hrf("spmg1")(sample_at), dtype=np.float64).reshape(
        -1
    )

    def _rmse(fit: HrfEstimate) -> float:
        errors = []
        for curve_i, row in fit.curve_info.iterrows():
            condition = str(row.condition).rsplit(".", 1)[-1]
            truth_row = 0 if condition.endswith("A") else 1
            for voxel in range(amplitudes.shape[1]):
                truth = truth_shape * amplitudes[truth_row, voxel]
                errors.append(
                    np.sqrt(np.mean((fit.estimate[:, curve_i, voxel] - truth) ** 2))
                )
        return float(np.mean(errors))

    gcv = estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=dataset,
        block="run",
        rsam=sample_at,
        k=8,
        lam="gcv",
        lam_grid=[0.0, 0.1, 1.0, 10.0, 100.0],
    )
    unpenalized = estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=dataset,
        block="run",
        rsam=sample_at,
        k=8,
        lam=0.0,
    )
    assert gcv.lam > 0.0
    assert gcv.lam != unpenalized.lam
    assert _rmse(gcv) < _rmse(unpenalized)


def test_tidy_and_predict_preserve_curve_labels():
    dataset, _amplitudes, _events = _make_spmg1_dataset(n_voxels=2, noise_sd=0.02)
    fit = estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=dataset,
        block="run",
        rsam=np.arange(0.0, 13.0),
        k=6,
        lam=0.5,
    )
    tbl = fit.tidy()
    assert list(tbl.columns) == [
        "time",
        "curve",
        "term",
        "condition",
        "voxel",
        "estimate",
        "std.error",
        "lower",
        "upper",
    ]
    assert len(tbl) == fit.time.size * len(fit.curves) * len(fit.voxels)
    assert set(tbl["curve"]) == set(fit.curves)
    assert tbl["term"].notna().all()
    assert tbl["condition"].notna().all()

    predicted = fit.predict(fit.time)
    assert predicted.shape == fit.estimate.shape
    np.testing.assert_allclose(predicted, fit.estimate, atol=1e-10)
    first = fit.as_matrix(curve=fit.curves[0])
    assert first.shape == (fit.time.size, len(fit.voxels))


def test_formula_path_is_not_the_lstsq_compat_helper():
    dataset, _amplitudes, _events = _make_spmg1_dataset(n_voxels=1, noise_sd=0.01)
    fit = fm.estimate_hrf(
        form="onset ~ hrf(condition)",
        dataset=dataset,
        block="run",
        rsam=np.arange(0.0, 13.0),
        k=6,
        lam=1.0,
    )
    assert isinstance(fit, HrfEstimate)
    assert fit.diagnostics["algorithm"] == "shared penalized multiresponse solve"


def test_trialwise_formula_is_rejected():
    dataset, _amplitudes, _events = _make_spmg1_dataset(n_voxels=1)
    with pytest.raises(ValueError, match="trialwise"):
        estimate_hrf(
            form="onset ~ trialwise()",
            dataset=dataset,
            block="run",
            rsam=np.arange(0.0, 8.0),
            k=6,
            lam=1.0,
        )
